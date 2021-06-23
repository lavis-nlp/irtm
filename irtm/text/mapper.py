# -*- coding: utf-8 -*-

import irt
from irt.data.pytorch import load_tokenizer

import irtm
from irtm.kgc import keen
from irtm.kgc import evaluator as kgc_evaluator
from irtm.text.config import Config

import gc
import yaml
import math
import torch
import torch.optim
from torch import nn
import transformers as tf
import pytorch_lightning as pl

import logging
from itertools import count
from itertools import repeat
from itertools import groupby
from itertools import zip_longest
from functools import partial
from dataclasses import dataclass
from collections import defaultdict

from typing import Dict
from typing import Tuple
from typing import Sequence


log = logging.getLogger(__name__)

TQDM_KWARGS = dict(
    position=2,
    ncols=80,
    leave=False,
    disable=True,
)


# --------------------


@dataclass
class UpstreamModels:

    tokenizer: tf.BertTokenizer
    kgc_model: keen.Model
    text_encoder: tf.BertModel

    @classmethod
    def load(K, config: Config, dataset: irt.Dataset):
        assert str(config.dataset) == str(dataset.path)

        text_encoder = tf.BertModel.from_pretrained(
            config.text_encoder,
            cache_dir=irtm.ENV.CACHE_DIR / "lib.transformers",
        )

        tokenizer = load_tokenizer(
            model_name=config.text_encoder,
            dataset_path=config.dataset,
        )

        log.info(f"resizing token embeddings to {len(tokenizer)}")
        text_encoder.resize_token_embeddings(len(tokenizer))

        kgc_model = keen.Model.load(
            path=config.kgc_model,
            dataset=dataset,
        )

        return K(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            kgc_model=kgc_model,
        )


# --------------------


class Base(nn.Module):

    registered = defaultdict(dict)

    @dataclass
    class Config:
        pass

    # to be implemented
    # name = 'name of the module'

    @property
    def config(self):
        return self._config

    def __init__(self, *args, config: Config = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config

    def __str__(self) -> str:
        return self.__name__

    # decorater
    @classmethod
    def module(Child, Impl):
        try:
            Impl.name
        except AttributeError:
            msg = "Class {Impl} has no attribute .name"
            raise irtm.IRTMError(msg)

        Base.registered[Child.__name__][Impl.name] = Impl
        return Impl

    @classmethod
    def init(Child, *, name: str = None, **kwargs):

        try:
            if name is None:
                name = "noop"

            A = Base.registered[Child.__name__][name]
        except KeyError:
            dicrep = yaml.dump(Base.registered, default_flow_style=False)

            msg = (
                f'could not find module "{name}"\n\n'
                f"available modules:\n"
                f"{dicrep}"
            )

            raise irtm.IRTMError(msg)

        config = A.Config(**kwargs)

        log.info(f"! initializing {A.__name__} with {config}")
        return A(config=config)


# --- AGGREGATION


class Aggregator(Base):
    pass


@Aggregator.module
class MaxPoolingAggregator_1(Aggregator):

    name = "max 1"

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.max(axis=1).values


@Aggregator.module
class MaxPoolingDropoutAggregator_1(Aggregator):

    name = "max dropout 1"

    @dataclass
    class Config(Base.Config):

        p: float

    def __init__(self, *args, config: "MaxPoolingDropoutAggregator_1.Config", **kwargs):

        super().__init__(*args, config=config, **kwargs)
        self.dropout = nn.Dropout(config.p)

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dropout(X.max(axis=1).values)


@Aggregator.module
class CLSAggregator_1(Aggregator):

    name = "cls 1"

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[:, 0]


@Aggregator.module
class CLSDropoutAggregator_1(Aggregator):

    name = "cls dropout 1"

    @dataclass
    class Config(Base.Config):

        p: float

    def __init__(self, *args, config: "CLSDropoutAggregator_1.Config", **kwargs):

        super().__init__(*args, config=config, **kwargs)
        self.dropout = nn.Dropout(config.p)

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dropout(X[:, 0])


# --- CONTEXT MERGES


class Reductor(Base):
    """
    Might reduce whole context to single representation
    """

    pass


@Reductor.module
class IndependentReductor_1(Reductor):

    name = "independent 1"

    def forward(self, entities: Tuple[int], context: torch.Tensor):
        return entities, context


@Reductor.module
class MaxReductor_1(Reductor):

    name = "max 1"

    def forward(self, entities: Tuple[int], context: torch.Tensor):
        # inner-batch indexes
        counter = count()

        # e.g. given two entities and a batch_size of 5:
        # (8, 8, 8, 7, 7) -> [(8, [0, 1, 2]), (7, [3, 4])]
        grouped = [
            (entity, [next(counter) for _ in grouper])
            for entity, grouper in groupby(entities)
        ]

        # batch x kge_dims -> unique entities x kge_dims
        # black formats this rather strangely
        # fmt: off
        pooled = torch.vstack(tuple(
            context[idxs].max(axis=0).values
            for _, idxs in grouped
        ))
        # fmt: on

        unique_entities = tuple(zip(*grouped))[0]
        return unique_entities, pooled


# --- PROJECTION


class Projector(Base):
    """
    Map a text context into KGE-space
    """

    pass


@Projector.module
class AffineProjector_1(Projector):
    """
    Ax + b
    """

    name = "affine 1"

    @dataclass
    class Config(Base.Config):

        input_dims: int
        output_dims: int

    # ---

    def __init__(self, *args, config: "AffineProjector_1.Config", **kwargs):

        super().__init__(*args, config=config, **kwargs)
        self.projector = nn.Linear(config.input_dims, config.output_dims)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.projector(X)


@Projector.module
class MLPProjector_1(Projector):
    """
    One hidden layer, ReLU for hidden and tanh for output
    """

    name = "mlp 1"

    @dataclass
    class Config(Base.Config):

        input_dims: int
        hidden_dims: int
        output_dims: int

    # ---

    def __init__(self, *args, config: "MLPProjector_1.Config", **kwargs):

        super().__init__(*args, config=config, **kwargs)

        self.projector = nn.Sequential(
            nn.Linear(config.input_dims, config.hidden_dims),
            nn.Tanh(),
            nn.Linear(config.hidden_dims, config.output_dims),
        )

    # batch x text_dims -> batch x kge_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.projector(X)


# --- COMPARING


class Comparator(Base):
    pass


@Comparator.module
class EuclideanComparator_1(Comparator):

    name = "euclidean 1"

    def forward(self, X, Y):
        return torch.dist(X, Y, p=2) / X.shape[0]


# --- NOOP


# used as drop in if no model.forward is needed
@Aggregator.module
@Projector.module
@Comparator.module
class Noop(Base):

    name = "noop"

    def forward(self, *args, **kwargs):
        assert False, "noop cannot forward"


# --- WIRING


OPTIMIZER = {
    "adam": torch.optim.Adam,
}


# unfortunately tf does not allow kwargs
# and the args order needs to be defined explicitly
SCHEDULER = {
    "constant": (tf.get_constant_schedule, ()),
    "constant with warmup": (
        tf.get_constant_schedule_with_warmup,
        ("num_warmup_steps"),
    ),
    "cosine with warmup": (
        tf.get_cosine_schedule_with_warmup,
        ("num_warmup_steps", "num_training_steps"),
    ),
    "cosine with hard restarts with warmup": (
        tf.get_cosine_with_hard_restarts_schedule_with_warmup,
        ("num_warmup_steps", "num_training_steps", "num_cycles"),
    ),
    "linear with warmup": (
        tf.get_linear_schedule_with_warmup,
        ("num_warmup_steps", "num_training_steps"),
    ),
}


@dataclass
class Components:

    config: Config

    # text encoder
    text_encoder: tf.BertModel

    # takes token representations and maps them to a single vector
    aggregator: Aggregator

    # takes sentence representations and (maybe) reduce them per entity
    reductor: Reductor

    # takes the context representation(s) and maps them to KGE space
    projector: Projector

    # compares the projected text embeddings to the target kg embeddings
    comparator: Comparator

    # the projection target
    kgc_model: keen.Model

    @property
    def kgc_model_name(self) -> str:
        return self.kgc_model.config.model.cls.lower()

    @classmethod
    def create(K, config: Config, upstream: UpstreamModels):
        aggregator = Aggregator.init(
            name=config.aggregator,
            **config.aggregator_args,
        )

        reductor = Reductor.init(
            name=config.reductor,
            **config.reductor_args,
        )

        projector = Projector.init(
            name=config.projector,
            **config.projector_args,
        )

        comparator = Comparator.init(
            name=config.comparator,
            **config.comparator_args,
        )

        if all((config.optimizer is not None, config.optimizer not in OPTIMIZER)):
            raise irtm.IRTMError(f'unknown optimizer "{config.optimizer}"')

        self = K(
            config=config,
            text_encoder=upstream.text_encoder,
            aggregator=aggregator,
            reductor=reductor,
            projector=projector,
            comparator=comparator,
            kgc_model=upstream.kgc_model,
        )

        return self


class Mapper(pl.LightningModule):
    """

    Maps textual descriptions to knowledge graph embeddings

    Open World Knowledge Graph Completion
    -------------------------------------

    Pykeen is used for evaluation, but it is not able to work with
    unknown entities. The following "hacks" are used to make it work:

    Pykeen TriplesFactories contain an E-sized entity-id, and R-sized
    relation-id mapping.  Entities and relations are combined a N x 3
    matrix (N = #triples). Each pykeen model uses the enitites and
    relations to construct torch.nn.Embdding (pykeen.nn.emb.Embdding)
    instances for relations and entities (E x D) and (R x D) (for
    pykeen.models.base.EntityRelationEmbeddingModel).

    After each validation epoch (see Mapper.on_validation_epoch_end)
    the internal entity embeddings are overwritten for (1) all
    entities (transductive) and (2) all owe entities (inductive) and
    the pykeen evaluator is used to quantify the kgc performance of
    the mapped entity representations.

    """

    irtmc: Components
    irtmod: irt.TorchModule

    # set in on_fit_start():
    train_subbatch_size: int
    val_subbatch_size: int
    ow_validation_batches: int

    # implemented own batch aggregation
    automatic_optimization: bool = False

    @property
    def debug(self) -> bool:
        try:
            return self.trainer.fast_dev_run
        except AttributeError:
            return self._debug

    @debug.setter
    def debug(self, val):
        assert not self.trainer, "use lightning to control debug"
        assert val is True or val is False
        self._debug = val

    def __init__(
        self,
        irtmod: irt.TorchModule,
        irtmc: Components,
        freeze_text_encoder: bool = False,
    ):
        super().__init__()

        # overwritten by self.trainer.fast_dev_run if present
        self._debug = False

        # properties

        self.irtmc = irtmc
        self.irtmod = irtmod

        # parameters

        self.encode = irtmc.text_encoder
        self.aggregate = irtmc.aggregator
        self.reduce = irtmc.reductor
        self.project = irtmc.projector
        self.loss = irtmc.comparator

        if freeze_text_encoder:
            log.info("! freezing text encoder")
            self.encode.requires_grad_(False)

        log.info("! freezing kgc model")
        self.keen = irtmc.kgc_model.keen
        self.keen.requires_grad_(False)

        # -- projections

        shape = (
            len(self.irtmod.kow.irt2keen),
            self.keen.entity_embeddings.embedding_dim,
        )

        log.info(f'register "projections" buffer of shape {shape}')

        # the projections buffer uses the pykeen indexes
        # (see _update_projections)

        self.register_buffer(
            "projections",
            torch.zeros(shape, requires_grad=False),
        )
        self.register_buffer(
            "projections_counts",
            torch.zeros(
                shape[0],
                requires_grad=False,
            ),
        )

        self.init_projections()

    def configure_optimizers(self):
        config = self.irtmc.config

        optimizer = OPTIMIZER[config.optimizer](
            self.parameters(), **config.optimizer_args
        )

        scheduler_name = config.scheduler or "constant"
        fn, kwargs = SCHEDULER[scheduler_name]

        last_epoch = self.current_epoch - 1
        args = [config.scheduler_args[k] for k in kwargs] + [last_epoch]
        scheduler = fn(optimizer, *args)

        log.info(f"initialized optimizer with {config.optimizer_args}")
        log.info(f"initialized {scheduler_name} scheduler with {kwargs=}")
        return [optimizer], [scheduler]

    #
    #   SELF REALISATION
    #

    def init_projections(self):
        """
        Initialize projection buffer

        This needs to be run before every dataloader iteration.
        After text samples have been provided by calling forward(),
        they need to reduced by invoking gather_projections().

        (!) Indexes used for projections are the pykeen entity indexes.
        A mapping of irtm indexes to pykeen indexes is given by
        self.irtmod.kow.irt2keen.

        """
        log.info("clearing projections buffer")

        self.projections.zero_()
        self.projections_counts.zero_()
        self._gathered_projections = False
        self._stats_projections = 0

    def gather_projections(self):
        log.info(f"gathered {int(self.projections_counts.sum().item())} projections")

        # calculate averages over all projections
        mask = self.projections_counts != 0
        self.projections[mask] /= self.projections_counts.unsqueeze(1)[mask]

        # TODO assert this reflect context counts of datasets
        self._gathered_projections = True

    def _update_projections(self, entities, projected):
        for e, v in zip(entities, projected.detach()):
            idx = self.irtmod.kow.irt2keen[e]
            self.projections[idx] += v
            self.projections_counts[idx] += 1

    # ---

    def forward_context(self, context: torch.Tensor):
        # mask padding and [MASK] tokens

        mask = 103  # TODO BERT specific!
        attention_mask = (context > 0) | (context == mask)
        attention_mask = attention_mask.to(dtype=torch.long)

        Y = self.encode(input_ids=context, attention_mask=attention_mask)
        return Y[0]

    def kge(self, entities: Tuple[int]):
        assert set(entities).issubset(self.irtmod.kow.dataset.split.closed_world.owe)

        mapped = tuple(map(lambda e: self.irtmod.kow.irt2keen[e], entities))
        embeddings = self.irtmc.kgc_model.embeddings(
            entities=mapped,
            device=self.device,
        )

        return embeddings

    def forward_subbatch(
        self,
        ents: Tuple[int],
        ctxs: torch.Tensor,
    ) -> Tuple[Tuple[int], torch.Tensor]:
        # sub-batch x tokens x text_dims
        encoded = self.forward_context(ctxs)

        # sub-batch x text_dims
        aggregated = self.aggregate(encoded)

        # (unique) entities x text_dims
        entities, reduced = self.reduce(ents, aggregated)

        # (unique) entities x kge_dims
        projected = self.project(reduced)

        # update projections
        self._update_projections(entities=entities, projected=projected)
        return entities, projected

    def forward(
        self,
        *,
        # entity -> [s1, s2, ...]
        batch: Tuple[Tuple[int], torch.Tensor],  # batch
        subbatch_size: int,
        optimize: bool = False,
        calculate_loss: bool = False,
    ):

        # _ts = datetime.now()
        # _ts_last = _ts

        # def timing(name):
        #     nonlocal _ts_last
        #     new = datetime.now()
        #     log.info(f"! {new - _ts} {new - _ts_last} {name}")
        #     _ts_last = new

        if optimize:
            assert calculate_loss
            optimizer = self.optimizers()

        ents, ctxs = batch

        # return values
        losses = [] if calculate_loss else None
        loss, ret = None, {}

        # sub-batching
        N = len(ents)
        subbatch_size = subbatch_size or len(ents)
        steps = range(0, N, subbatch_size)

        # timing("pre-subbatch")
        subbatches = list(zip_longest(steps, steps[1:], fillvalue=N))
        for j, k in subbatches:
            # timing(f"[{j}:{k}] enter loop")

            entities, projected = self.forward_subbatch(ents=ents[j:k], ctxs=ctxs[j:k])

            # timing(f"[{j}:{k}] forward")
            # entities = ents[j:k]
            # projected = torch.rand((len(entities), 500))
            # projected = projected.to(device=self.device)
            # projected.requires_grad = calculate_loss

            if calculate_loss:
                targets = self.kge(entities)
                loss = self.loss(projected, targets)
                losses.append(loss)
                # timing(f"[{j}:{k}] loss")

            if optimize:
                self.manual_backward(loss / len(subbatches))
                # timing(f"[{j}:{k}] backward")

                clip_val = self.irtmc.config.clip_val
                if clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        max_norm=float(clip_val),
                        norm_type=2.0,
                    )

            ret.update(
                {
                    e: torch.vstack((ret[e].detach(), x)) if e in ret else x
                    for e, x in zip(entities, projected)
                }
            )

            # timing(f"[{j}:{k}] ret update")

        if optimize:
            optimizer.step()
            optimizer.zero_grad()
            # timing("optimizer step")

        if calculate_loss:
            loss = torch.stack(losses).mean()
            # timing("loss calculation")

        if calculate_loss:
            return loss, ret
        else:
            assert loss is None
            return ret

    #
    #   TRAINING
    #

    def training_step(
        self,
        batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int,
    ):
        # batch, batch x kge_dims
        loss, projections = self.forward(
            batch=batch,
            optimize=True,
            calculate_loss=True,
            subbatch_size=self.train_subbatch_size,
        )

        self.log("train_loss_step", loss)
        return loss

    #
    #   VALIDATION
    #

    def validation_step(
        self,
        batch: Sequence[Tuple[int, torch.Tensor]],
        batch_idx: int,
        *args,
    ):
        # it updates the projections buffer
        self.forward(
            batch=batch,
            subbatch_size=self.val_subbatch_size,
        )

    def validation_epoch_end(self, *_):
        self._run_kgc_evaluations()

    def run_kgc_evaluation(self, kind: str, factory):  # pykeen CoreTriplesFactory
        assert self._gathered_projections, "run gather_projections()"

        mapped_triples = factory.mapped_triples
        if self.debug:
            mapped_triples = mapped_triples[:100]  # choice arbitrary

        kow: irt.KeenOpenWorld = self.irtmod.kow

        if kind == "transductive":
            entities = kow.dataset.split.closed_world.owe
            filtered_triples = []

        elif kind == "inductive":
            entities = kow.dataset.split.open_world_valid.owe
            filtered_triples = [kow.closed_world.mapped_triples]

        elif kind == "test":
            entities = kow.dataset.split.open_world_test.owe
            filtered_triples = [
                kow.closed_world.mapped_triples,
                kow.open_world_valid.mapped_triples,
            ]

        else:
            raise irtm.IRTMError(f"unknown kind: {kind}")

        # map to pykeen indexes
        entities = torch.tensor([self.irtmod.kow.irt2keen[e] for e in entities])

        log.info(
            f"running {kind} evaluation"
            f" with {mapped_triples.shape[0]} triples"
            f" replacing {len(entities)} embeddings"
        )

        # -- embedding shenanigans

        original = self.keen.entity_embeddings.cpu()

        new = self.keen.entity_embeddings.__class__.init_with_device(
            num_embeddings=len(kow.irt2keen),
            embedding_dim=self.keen.entity_embeddings.embedding_dim,
            device=self.device,
        )

        # copy trained closed world embeddings and overwrite with projections
        new._embeddings.weight.zero_()
        new._embeddings.weight[: original.num_embeddings] = original._embeddings.weight
        new._embeddings.weight[entities] = self.projections[entities]

        # plug in new embeddings
        self.keen.entity_embeddings = new

        evaluation_result = kgc_evaluator.evaluate(
            model=self.keen,
            config=self.irtmc.kgc_model.config,
            triples=mapped_triples,
            filtered_triples=filtered_triples,
            tqdm_kwargs=TQDM_KWARGS,
        )

        # restore
        del new
        self.keen.entity_embeddings = original.to(device=self.device)

        # # -- /embedding shenanigans

        metrics = evaluation_result.metrics
        metrics_map = {
            f"both.realistic.{k}": v
            for k, v in {
                "hits_at_1": "h@1",
                "hits_at_10": "h@10",
                "inverse_harmonic_mean_rank": "MRR",
            }.items()
        }

        log.info(
            f"finished {kind}: "
            + ", ".join([f"{metrics_map[k]}: {metrics[k]:.4f}" for k in metrics_map])
        )

        return metrics

    def _run_kgc_evaluations(self):
        if self.global_step == 0 and not self.debug:
            log.info("skipping kgc evaluation; logging phony kgc result")
            for kind in ("inductive", "transductive"):

                self._log_kgc_results(
                    kind=kind,
                    results=keen.mock_metric_results().to_flat_dict(),
                )

            return

        # --

        def _save_run(kind, factory, attempt: int = 0):
            if attempt > 5:
                raise irtm.IRTMError("ran out of patience")

            log.info(f"running kgc evaluation attempt {attempt}")

            try:
                torch.cuda.empty_cache()
                results = self.run_kgc_evaluation(kind=kind, factory=factory)
                self._log_kgc_results(kind=kind, results=results)

                log.info(f"! finished {kind} kgc evaluation on attempt {attempt}")
            except RuntimeError as exc:
                log.error(f"registered error: {exc}")
                torch.cuda.empty_cache()
                _save_run(kind=kind, factory=factory, attempt=attempt + 1)

        # --

        self.gather_projections()
        for kind, factory in (
            ("transductive", self.irtmod.kow.closed_world),
            ("inductive", self.irtmod.kow.open_world_valid),
        ):
            results = self.run_kgc_evaluation(kind=kind, factory=factory)
            self._log_kgc_results(kind=kind, results=results)
            # try:
            #     _save_run(kind=kind, factory=factory)

            # except irtm.IRTMError:
            #     log.error("skipping kgc evaluation in this epoch :(")
            #     self._log_kgc_results(kind=kind, results=keen.mock_metric_results())
            #     break

    def _log_kgc_results(
        self,
        kind: str = None,
        results: Dict = None,
    ):
        self.log_dict({f"{kind}/{k}": v for k, v in results.items()})

    def run_memcheck(self, test: bool = False):
        # removing some memory because some cuda
        # allocated memory is not visible
        device_properties = torch.cuda.get_device_properties(self.device)
        total_memory = int(device_properties.total_memory * 0.9)

        log.info(
            f"checking {device_properties.name} with around "
            f" ~{total_memory // 1024**3}GB"
        )

        if not test:
            trained = self.training
            self.train()

        loaders = (
            self.irtmod.train_dataloader(),
            self.irtmod.val_dataloader(),
            self.irtmod.test_dataloader(),
        )

        for loader in loaders:
            train = loader.dataset.name.endswith("train")  # TODO

            # moving some noise with the shape of the largest
            # possible sample through the model to detect oom problems

            log.info(
                f"checking {loader.dataset.name} (largest batch:"
                f" {loader.dataset.max_context_size})"
            )

            sample = loader.dataset[loader.dataset.max_context_idx]
            count = math.ceil(loader.subbatch_size / len(sample[1]))
            samples = repeat(sample, count)

            batch = loader.collate_fn(list(samples))
            batch = batch[0], batch[1].to(self.device)

            log.info(
                f"trying batch with {batch[1].shape}"
                f" subbatch size: {loader.subbatch_size}"
            )

            forward = partial(
                self.forward,
                batch=batch,
                optimize=train,
                calculate_loss=train,
                subbatch_size=loader.subbatch_size,
            )

            if not train:
                with torch.no_grad():
                    forward()
            else:
                forward()

            # can be > 100 sometimes...
            res_memory = torch.cuda.memory_reserved(self.device)
            memory_usage = (res_memory / total_memory) * 100

            log.info(
                f"! {loader.dataset.name} is using" f" ~{int(memory_usage)}% memory"
            )

            del batch
            gc.collect()
            torch.cuda.empty_cache()

        log.info("finished memory check")
        if not test and not trained:
            self.eval()

    #
    # HOOKS
    #

    def on_fit_start(self):
        self.train_subbatch_size = self.irtmod.train_dataloader().subbatch_size
        self.val_subbatch_size = self.irtmod.val_dataloader().subbatch_size

        log.info("running custom pre-training sanity check")
        self.run_memcheck()

    def on_train_epoch_start(self):
        log.info(
            f"! starting epoch {self.current_epoch}"
            f" (step={self.global_step});"
            f" running on {self.device}"
        )

        self.init_projections()

        _embs = self.keen.entity_embeddings._embeddings
        assert not _embs.weight.requires_grad

        _embs = self.keen.relation_embeddings._embeddings
        assert not _embs.weight.requires_grad
