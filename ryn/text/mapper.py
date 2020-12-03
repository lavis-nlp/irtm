# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import trainer as kgc_trainer
from ryn.text import data
from ryn.text.config import Config
from ryn.common import helper
from ryn.common import logging

import math
import torch
import torch.optim
from torch import nn
import transformers as tf
import pytorch_lightning as pl
import horovod.torch as hvd

# https://github.com/pykeen/pykeen/pull/132
# from pykeen.nn import emb as keen_emb

import yaml

from itertools import repeat
from dataclasses import dataclass
from collections import defaultdict


log = logging.get("text.mapper")

TQDM_KWARGS = dict(
    position=2,
    ncols=80,
    leave=False,
    disable=True,
)


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
            raise ryn.RynError(msg)

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

            raise ryn.RynError(msg)

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

    def __init__(
        self, *args, config: "MaxPoolingDropoutAggregator_1.Config", **kwargs
    ):

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

    def __init__(
        self, *args, config: "CLSDropoutAggregator_1.Config", **kwargs
    ):

        super().__init__(*args, config=config, **kwargs)
        self.dropout = nn.Dropout(config.p)

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dropout(X[:, 0])


# --- PROJECTION


class Projector(Base):
    pass


@Projector.module
class AffineProjector_1(Projector):

    name = "affine 1"

    @dataclass
    class Config(Base.Config):

        input_dims: int
        output_dims: int

    # ---

    def __init__(self, *args, config: "AffineProjector_1.Config", **kwargs):

        super().__init__(*args, config=config, **kwargs)
        self.projector = nn.Linear(config.input_dims, config.output_dims)

    # batch x text_dims -> batch x kge_dims
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

    # takes token representations and maps them to
    # a single vector for the projector
    aggregator: Aggregator

    # takes an aggeragated vector of text embeddings
    # and projects them to the kg embeddings
    projector: Projector

    # compares the projected text embeddings
    # to the target kg embeddings
    comparator: Comparator

    # the projection target
    kgc_model: keen.Model

    @property
    def kgc_model_name(self) -> str:
        return self.kgc_model.config.model.cls.lower()

    @classmethod
    @helper.notnone
    def create(K, *, config: Config = None, models: data.Models = None):
        aggregator = Aggregator.init(
            name=config.aggregator, **config.aggregator_args
        )

        projector = Projector.init(
            name=config.projector, **config.projector_args
        )

        comparator = Comparator.init(
            name=config.comparator, **config.comparator_args
        )

        if all(
            (config.optimizer is not None, config.optimizer not in OPTIMIZER)
        ):
            raise ryn.RynError(f'unknown optimizer "{config.optimizer}"')

        self = K(
            config=config,
            text_encoder=models.text_encoder,
            aggregator=aggregator,
            projector=projector,
            comparator=comparator,
            kgc_model=models.kgc_model,
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

    @property
    def rync(self) -> Components:
        return self._rync

    @property
    def data(self) -> data.DataModule:
        return self._data

    @property
    def projection_aggregation(self) -> str:
        return self._projection_aggregation

    @projection_aggregation.setter
    def projection_aggregation(self, projection_aggregation):
        if not any(
            (
                projection_aggregation == "max",
                projection_aggregation == "avg",
                projection_aggregation == "last",
            )
        ):
            raise ryn.RynError(
                "unknown projection aggregation:" f" {projection_aggregation}"
            )

        self._projection_aggregation = projection_aggregation

    def __init__(
        self,
        data=None,  # data.DataModule
        rync: Components = None,
        freeze_text_encoder: bool = False,
        # supported are: avg, max, last
        projection_aggregation: str = "avg",
    ):
        assert data is not None
        assert rync is not None

        super().__init__()

        # overwritten by self.trainer.fast_dev_run if present
        self._debug = False

        # properties

        self._data = data
        self._rync = rync
        self.projection_aggregation = projection_aggregation

        # parameters

        self.encode = rync.text_encoder
        self.aggregate = rync.aggregator
        self.project = rync.projector
        self.loss = rync.comparator

        if freeze_text_encoder:
            log.info("! freezing text encoder")
            self.encode.requires_grad_(False)

        self.keen = rync.kgc_model.keen
        rync.kgc_model.freeze()

        # -- projections

        shape = (
            len(self.data.kgc.ryn2keen),
            self.keen.entity_embeddings.embedding_dim,
        )

        log.info(f'register "projections" buffer of shape {shape}')

        self.register_buffer(
            "projections", torch.zeros(shape, requires_grad=False)
        )

        self.register_buffer(
            "projections_counts", torch.zeros(shape[0], requires_grad=False)
        )

        self.init_projections()

    def configure_optimizers(self):
        config = self.rync.config

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
        log.info("clearing projections buffer")
        self.projections.zero_()
        self.projections_counts.zero_()

    @helper.notnone
    def update_projections(self, entities=None, projected=None):
        for e, v in zip(entities, projected.detach()):
            idx = self.data.kgc.ryn2keen[e]

            if self.projection_aggregation == "max":
                self.projections[idx] = torch.max(self.projections[idx], v)
                self.projections_counts[idx] = 1

            if self.projection_aggregation == "avg":
                self.projections[idx] += v
                self.projections_counts[idx] += 1

            if self.projection_aggregation == "last":
                self.projections[idx] = v
                self.projections_counts = 1

    # ---

    def forward_sentences(self, sentences: torch.Tensor):
        # mask padding and [MASK] tokens

        mask = 103  # TODO BERT specific!
        attention_mask = (sentences > 0) | (sentences == mask)
        attention_mask = attention_mask.to(dtype=torch.long)

        return self.encode(input_ids=sentences, attention_mask=attention_mask)[
            0
        ]

    def forward_entities(self, entities: torch.Tensor):
        return self.rync.kgc_model.embeddings(
            entities=entities, device=self.device
        )

    @helper.notnone
    def forward(
        self,
        *,
        sentences: torch.Tensor = None,  # batch x tokens
    ):

        # batch x tokens x text_dims
        encoded = self.forward_sentences(sentences)

        # batch x text_dims
        aggregated = self.aggregate(encoded)

        # batch x kge_dims
        projected = self.project(aggregated)

        return projected

    #
    #   TRAINING
    #

    def training_step(self, batch, batch_idx: int):
        sentences, entities = batch

        # batch x kge_dims
        projected = self.forward(sentences=sentences)
        self.update_projections(entities=entities, projected=projected)

        # batch x kge_dims
        target = self.forward_entities(entities)
        loss = self.loss(projected, target)

        self.log_dict(
            dict(
                train_loss_step=loss,
            )
        )

        return loss

    #
    #   VALIDATION
    #

    @helper.notnone
    def _geometric_validation_step(
        self,
        sentences=None,
        entities=None,
        kind: str = None,
    ):

        # batch x kge_dims
        projected = self.forward(sentences=sentences)
        self.update_projections(entities=entities, projected=projected)

        # batch x kge_dims
        target = self.forward_entities(entities)

        # batch
        losses = self.loss(projected, target)

        # partition losses into inductive and transductive
        self.log_dict({f"{kind}.valid_loss_step": losses})

    @helper.notnone
    def _kgc_validation_step(
        self,
        sentences=None,
        entities=None,
        batch_idx=None,
    ):
        projected = self.forward(sentences=sentences)
        self.update_projections(entities=entities, projected=projected)
        if batch_idx == self._ow_validation_batches - 1:
            self._run_kgc_evaluations()

    def validation_step(self, batch, batch_idx: int, *args):
        dataloader_idx = args[0] if args else None  # nasty!
        sentences, entities = batch

        if self.data.should_evaluate_geometric(dataloader_idx):
            self._geometric_validation_step(
                sentences=sentences,
                entities=entities,
                kind=self.data.geometric_validation_kind(dataloader_idx),
            )
        elif self.data.should_evaluate_kgc(dataloader_idx):
            self._kgc_validation_step(
                sentences=sentences,
                entities=entities,
                batch_idx=batch_idx,
            )
        else:
            assert False

    @helper.notnone
    def run_kgc_evaluation(
        self, *, kind: str = None, triples=None
    ):  # data.Triples

        global TQDM_KWARGS
        if hvd.size() == 1:
            TQDM_KWARGS["disable"] = False

        assert kind in ["transductive", "inductive", "test"]

        log.info(
            f"running {kind} evaluation"
            f" with {triples.factory.mapped_triples.shape[0]} triples"
            f" replacing {len(triples.entities)} embeddings"
        )

        # -- embedding shenanigans

        Embedding = torch.nn.Embedding.from_pretrained
        original_weights = self.keen.entity_embeddings.weight.cpu()

        new_weights = torch.zeros(
            (
                len(self.data.kgc.ryn2keen),
                original_weights.shape[1],
            )
        ).to(self.device)

        new_weights[: original_weights.shape[0]] = original_weights
        idxs = list(map(lambda i: self.data.kgc.ryn2keen[i], triples.entities))

        new_weights[idxs] = self.projections[idxs]

        self.keen.entity_embeddings = Embedding(new_weights).to(self.device)

        mapped_triples = triples.factory.mapped_triples
        if self.debug:
            mapped_triples = mapped_triples[:100]  # choice arbitrary

        evaluation_result = kgc_trainer.evaluate(
            model=self.keen,
            config=self.rync.kgc_model.config,
            mapped_triples=mapped_triples,
            tqdm_kwargs=TQDM_KWARGS,
        )

        # restore original embeddings
        self.keen.entity_embeddings = Embedding(original_weights).to(
            self.device
        )

        # -- /embedding shenanigans

        def _result_dict(name=None, result=None):
            def _item(key, val):
                return f"{name}.{key}", torch.Tensor([val])

            return dict(
                (k, v.item())
                for k, v in (
                    _item(
                        "hits@10",
                        result.metrics["hits_at_k"]["both"]["avg"][10],
                    ),
                    _item(
                        "hits@5", result.metrics["hits_at_k"]["both"]["avg"][5]
                    ),
                    _item(
                        "hits@1", result.metrics["hits_at_k"]["both"]["avg"][1]
                    ),
                    _item("mr", result.metrics["mean_rank"]["both"]["avg"]),
                    _item(
                        "mrr",
                        result.metrics["mean_reciprocal_rank"]["both"]["avg"],
                    ),
                    _item("amr", result.metrics["adjusted_mean_rank"]["both"]),
                )
            )

        result_metrics = _result_dict(name=kind, result=evaluation_result)

        log.info(
            f"! finished {kind} evaluation with"
            f' {evaluation_result.metrics["hits_at_k"]["both"]["avg"][10]:2.3f} hits@10'  # noqa: E501
            f' and {evaluation_result.metrics["mean_reciprocal_rank"]["both"]["avg"]:2.3f} MRR'  # noqa: E501
        )

        return result_metrics

    def _run_kgc_evaluations(self):
        if self.global_step == 0 and not self.debug:
            log.info("skipping kgc evaluation; logging phony kgc result")
            self.log_dict(self._mock_kgc_results())
            return

        log.info(
            f"[{hvd.local_rank()}] gathered"
            f" {int(self.projections_counts.sum().item())}"
            " projections"
        )

        self.projections = hvd.allreduce(self.projections, op=hvd.Sum)

        self.projections_counts = hvd.allreduce(
            self.projections_counts, op=hvd.Sum
        )

        if hvd.local_rank() != 0:
            log.info(
                f"[{hvd.local_rank()}] servant skips kgc evaluation;"
                " logging phony kgc result"
            )

            self.log_dict(self._mock_kgc_results())
            return

        # calculate averages over all projections
        mask = self.projections_counts != 0
        self.projections[mask] /= self.projections_counts.unsqueeze(1)[mask]

        log.info(
            f"gathered {int(self.projections_counts.sum().item())}"
            " projections from processes"
        )

        # TODO assert this reflect context
        # counts of datasets (unless fast_dev_run)

        # --

        transductive_result = self.run_kgc_evaluation(
            kind="transductive", triples=self.data.kgc.transductive
        )
        inductive_result = self.run_kgc_evaluation(
            kind="inductive", triples=self.data.kgc.inductive
        )

        log.info("logging kgc result")
        self.log_dict(
            {
                **transductive_result,
                **inductive_result,
            }
        )

    def _mock_kgc_results(self):
        log.info("mocking kgc evaluation results")

        phony_results = {
            f"{name}.{key}": torch.Tensor([val])
            for name, triples in (
                ("inductive", self.data.kgc.inductive.factory.triples),
                ("transductive", self.data.kgc.transductive.factory.triples),
            )
            for key, val in (
                {
                    "hits@10": 0.0,
                    "hits@5": 0.0,
                    "hits@1": 0.0,
                    "mr": len(triples) // 2,
                    "mrr": 0.0,
                    "amr": 1.0,
                }
            ).items()
        }

        return phony_results

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
            [self.data.train_dataloader()]
            + self.data.val_dataloader()
            + [self.data.test_dataloader()]
        )

        for loader in loaders:
            # moving some noise with the shape of the largest
            # possible sample through the model to detect oom problems

            log.info(
                f"checking {loader.dataset.name} (max sequence length:"
                f" {loader.dataset.max_sequence_length})"
            )

            sample = loader.dataset[loader.dataset.max_sequence_idx]
            batch = repeat(sample, loader.batch_size)
            sentences, _ = loader.collate_fn(batch)

            log.info(f"trying batch with {sentences.shape}")

            sentences = sentences.to(self.device)
            self.forward(sentences=sentences)

            res_memory = torch.cuda.memory_reserved(self.device)
            memory_usage = (res_memory / total_memory) * 100
            log.info(
                f"! {loader.dataset.name} is using"
                f" ~{int(memory_usage)}% memory"
            )

            del sentences
            torch.cuda.empty_cache()

        log.info("finished memory check")
        if not test and not trained:
            self.eval()

    #
    # HOOKS
    #

    def on_fit_start(self):

        # hvd is initialized now

        self._ow_validation_batches = math.ceil(
            len(self.data.kgc_dataloader) / hvd.size()
        )

        if hvd.size() != 1 and self.projection_aggregation == "max":
            raise ryn.RynError(
                "max pooling is not supported for multi-gpu setups"
            )

        # --

        log.info("running custom pre-training sanity check")
        self.run_memcheck()

    def on_train_epoch_start(self):
        log.info(
            f"! starting epoch {self.current_epoch}"
            f" (step={self.global_step});"
            f" running on {self.device}"
        )

        self.init_projections()

        assert not self.keen.entity_embeddings.weight.requires_grad
        assert not self.keen.relation_embeddings.weight.requires_grad
