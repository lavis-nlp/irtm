# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import trainer as kgc_trainer
from ryn.text import data
from ryn.text.config import Config
from ryn.common import helper
from ryn.common import logging

import torch as t
import torch.optim
from torch import nn
import transformers as tf
import pytorch_lightning as pl
import horovod.torch as hvd

# https://github.com/pykeen/pykeen/pull/132
# from pykeen.nn import emb as keen_emb

import yaml
from tqdm import tqdm

from itertools import repeat
from dataclasses import dataclass
from collections import defaultdict

from typing import Any
from typing import Dict


log = logging.get('text.mapper')

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
            msg = 'Class {Impl} has no attribute .name'
            raise ryn.RynError(msg)

        Base.registered[Child.__name__][Impl.name] = Impl
        return Impl

    @classmethod
    def init(Child, *, name: str = None, **kwargs):

        try:
            A = Base.registered[Child.__name__][name]
        except KeyError:
            dicrep = yaml.dump(Base.registered, default_flow_style=False)

            msg = (
                f'could not find module "{name}"\n\n'
                f'available modules:\n'
                f'{dicrep}')

            raise ryn.RynError(msg)

        config = A.Config(**kwargs)

        log.info(f'! initializing {A.__name__} with {config}')
        return A(config=config)


# --- AGGREGATION

class Aggregator(Base):
    pass


@Aggregator.module
class MaxPoolingAggregator_1(Aggregator):

    name = 'max 1'

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.max(axis=1).values


@Aggregator.module
class MaxPoolingDropoutAggregator_1(Aggregator):

    name = 'max dropout 1'

    @dataclass
    class Config(Base.Config):

        p: float

    def __init__(
            self, *args,
            config: 'MaxPoolingDropoutAggregator_1.Config',
            **kwargs):

        super().__init__(*args, config=config, **kwargs)
        self.dropout = nn.Dropout(config.p)

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dropout(X.max(axis=1).values)


@Aggregator.module
class CLSAggregator_1(Aggregator):

    name = 'cls 1'

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[:, 0]


@Aggregator.module
class CLSDropoutAggregator_1(Aggregator):

    name = 'cls dropout 1'

    @dataclass
    class Config(Base.Config):

        p: float

    def __init__(
            self, *args,
            config: 'CLSDropoutAggregator_1.Config',
            **kwargs):

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

    name = 'affine 1'

    @dataclass
    class Config(Base.Config):

        input_dims: int
        output_dims: int

    # ---

    def __init__(
            self, *args,
            config: 'AffineProjector_1.Config',
            **kwargs):

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

    name = 'mlp 1'

    @dataclass
    class Config(Base.Config):

        input_dims: int
        hidden_dims: int
        output_dims: int

    # ---

    def __init__(
            self, *args,
            config: 'MLPProjector_1.Config',
            **kwargs):

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

    name = 'euclidean 1'

    def forward(self, X, Y):
        return torch.dist(X, Y, p=2) / X.shape[0]


# --- WIRING


OPTIMIZER = {
    'adam': torch.optim.Adam,
}


@dataclass
class Components:

    Optimizer: t.optim.Optimizer
    optimizer_args: Dict[str, Any]

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
            name=config.aggregator,
            **config.aggregator_args)

        projector = Projector.init(
            name=config.projector,
            **config.projector_args)

        comparator = Comparator.init(
            name=config.comparator,
            **config.comparator_args)

        self = K(
            Optimizer=OPTIMIZER[config.optimizer],
            optimizer_args=config.optimizer_args,
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
    def lr(self):
        return self._lr

    @lr.setter  # required for auto_lr_find
    def lr(self, val):
        self._lr = val

    @property
    def rync(self) -> Components:
        return self._rync

    @property
    def datasets(self) -> data.Datasets:
        return self._datasets

    @helper.notnone
    def __init__(
            self,
            rync: Components = None,
            datasets: data.Datasets = None,
            freeze_text_encoder: bool = False):

        super().__init__()

        # this flag exists to prevent the kgc
        # evaluation to be run after the model
        # was restored from a checkpoint
        self._has_trained = False

        # properties

        self._rync = rync
        self._lr = self.rync.optimizer_args['lr']
        self._datasets = datasets

        # parameters

        self.encode = rync.text_encoder
        self.aggregate = rync.aggregator
        self.project = rync.projector
        self.loss = rync.comparator

        if freeze_text_encoder:
            log.info('freezing text encoder')
            raise NotImplementedError()

        self.keen = rync.kgc_model.keen
        rync.kgc_model.freeze()

        # buffer

        # this is a entities x kgc_dims buffer to
        # accumulate projections throughout training and
        # validation; they are later used for kgc validation
        shape = (
            len(self.datasets.ryn2keen),
            self.keen.entity_embeddings.embedding_dim,
        )

        log.info(f'register "projections" buffer of shape {shape}')

        self.register_buffer(
            'projections',
            torch.zeros(shape, requires_grad=False))

        self.register_buffer(
            'projections_counts',
            torch.zeros(shape[0], requires_grad=False))

        self.init_projections()

    def configure_optimizers(self):
        optim = self.rync.Optimizer(
            self.parameters(),
            **self.rync.optimizer_args)

        log.info(f'initialized optimizer with {self.rync.optimizer_args}')
        return optim

    #
    #   SELF REALISATION
    #

    def init_projections(self):
        log.info('clearing projections buffer')
        self.projections.zero_()
        self.projections_counts.zero_()

    @helper.notnone
    def update_projections(self, entities=None, projected=None):
        for e, v in zip(entities, projected.detach()):
            idx = self.datasets.ryn2keen[e]
            self.projections[idx] += v
            self.projections_counts[idx] += 1

    # ---

    def forward_sentences(self, sentences: torch.Tensor):
        # mask padding and [MASK] tokens
        # mask = self.rync.tokenizer.base.vocab['[MASK]']
        attention_mask = (sentences > 0)  # | (sentences == mask)
        attention_mask = attention_mask.to(dtype=torch.long)

        return self.encode(
            input_ids=sentences,
            attention_mask=attention_mask)[0]

    def forward_entities(self, entities: torch.Tensor):
        return self.rync.kgc_model.embeddings(
            entities=entities,
            device=self.device)

    @helper.notnone
    def forward(
            self, *,
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
        self.log('train_loss_step', loss)

        return loss

    #
    #   VALIDATION
    #

    def validation_step(self, batch, batch_idx: int):
        sentences, entities = batch

        # batch x kge_dims
        projected = self.forward(sentences=sentences)
        self.update_projections(entities=entities, projected=projected)

        # batch x kge_dims
        target = self.forward_entities(entities)

        loss = self.loss(projected, target)
        self.log('valid_loss_step', loss)
        self.log_dict(self._last_kgc_metrics)

        return loss

    @helper.notnone
    def run_kgc_evaluation(
            self, *,
            kind: str = None,
            triples: data.Triples = None):

        global TQDM_KWARGS
        if hvd.size == 1:
            TQDM_KWARGS['disable'] = False

        assert kind in ['transductive', 'inductive', 'test']

        log.info(
            f'running {kind} evaluation'
            f' with {triples.factory.mapped_triples.shape[0]} triples'
            f' replacing {len(triples.entities)} embeddings'
        )

        # -- embedding shenanigans

        Embedding = torch.nn.Embedding.from_pretrained
        original_weights = self.keen.entity_embeddings.weight.cpu()

        new_weights = torch.zeros((
            len(self.datasets.ryn2keen),
            original_weights.shape[1],
        )).to(self.device)

        new_weights[:original_weights.shape[0]] = original_weights
        idxs = list(map(lambda i: self.datasets.ryn2keen[i], triples.entities))
        new_weights[idxs] = self.projections[idxs]

        self.keen.entity_embeddings = Embedding(
            new_weights).to(self.device)

        mapped_triples = triples.factory.mapped_triples
        if self.trainer.fast_dev_run:
            mapped_triples = mapped_triples[:100]  # choice arbitrary

        evaluation_result = kgc_trainer.evaluate(
            model=self.keen,
            config=self.rync.kgc_model.config,
            mapped_triples=mapped_triples,
            tqdm_kwargs=TQDM_KWARGS,
        )

        # restore original embeddings
        self.keen.entity_embeddings = Embedding(
            original_weights).to(self.device)

        # -- /embedding shenanigans

        def _result_dict(name=None, result=None):

            def _item(key, val):
                return f'{name}.{key}', torch.Tensor([val])

            return dict((k, v.item()) for k, v in (
                _item('hits@10',
                      result.metrics['hits_at_k']['both']['avg'][10]),
                _item('hits@5',
                      result.metrics['hits_at_k']['both']['avg'][5]),
                _item('hits@1',
                      result.metrics['hits_at_k']['both']['avg'][1]),
                _item('mr',
                      result.metrics['mean_rank']['both']['avg']),
                _item('mrr',
                      result.metrics['mean_reciprocal_rank']['both']['avg']),
                _item('amr',
                      result.metrics['adjusted_mean_rank']['both']),
            ))

        result_metrics = _result_dict(
            name=kind,
            result=evaluation_result
        )

        log.info(
            f'! finished {kind} evaluation with'
            f' {evaluation_result.metrics["hits_at_k"]["both"]["avg"][10]:2.3f} hits@10'  # noqa: E501
            f' and {evaluation_result.metrics["mean_reciprocal_rank"]["both"]["avg"]:2.3f} MRR')  # noqa: E501

        return result_metrics

    def _run_kgc_evaluations(self):
        assert hvd.local_rank() == 0

        # TODO assert id mappings are equal for cw entities
        # and relations (self.rync.kgc_model.keen_dataset.training)
        log.info('running kgc evaluations')

        # generate projections for the inductive scenario
        gen = self.datasets.text_inductive
        for (sentences, entities) in tqdm(gen, **TQDM_KWARGS):
            projected = self.forward(sentences=sentences.to(self.device))
            self.update_projections(entities=entities, projected=projected)

            if self.trainer.fast_dev_run:
                break

        # calculate averages over all projections
        mask = self.projections_counts != 0
        self.projections[mask] /= self.projections_counts.unsqueeze(1)[mask]

        log.info(f'gathered {int(self.projections_counts.sum().item())}'
                 ' projections from processes')

        # TODO assert this reflect context
        # counts of datasets (unless fast_dev_run)

        # --

        transductive_result = self.run_kgc_evaluation(
            kind='transductive', triples=self.datasets.kgc_transductive)
        inductive_result = self.run_kgc_evaluation(
            kind='inductive', triples=self.datasets.kgc_inductive)

        log.info('updating kgc metrics')
        self._last_kgc_metrics = {
            **transductive_result,
            **inductive_result,
        }

    def _mock_kgc_results(self):
        log.info('mocking kgc evaluation results')

        phony_results = {
            f'{name}.{key}': torch.Tensor([val])
            for name, triples in (
                    ('inductive',
                     self.datasets.kgc_inductive.factory.triples),
                    ('transductive',
                     self.datasets.kgc_transductive.factory.triples))
            for key, val in ({
                    'hits@10': 0.0,
                    'hits@5': 0.0,
                    'hits@1': 0.0,
                    'mr': len(triples) // 2,
                    'mrr': 0.0,
                    'amr': 1.0,
            }).items()
        }

        return phony_results

    def run_memcheck(self, test: bool = False):
        # removing some memory because some cuda
        # allocated memory is not visible
        device_properties = torch.cuda.get_device_properties(self.device)
        total_memory = int(device_properties.total_memory * 0.9)
        log.info(
            f'checking {device_properties.name} with around '
            f' ~{total_memory // 1024**3}GB')

        if not test:
            trained = self.training
            self.train()

        loaders = [
            self.datasets.text_train,
            self.datasets.text_valid,
            self.datasets.text_inductive,
        ]

        if test:
            loaders.append(self.datasets.text_test)

        for loader in loaders:
            # moving some noise with the shape of the largest
            # possible sample through the model to detect oom problems

            log.info(
                f'checking {loader.dataset.name} (max sequence length:'
                f' {loader.dataset.max_sequence_length})')

            sample = loader.dataset[loader.dataset.max_sequence_idx]
            batch = repeat(sample, loader.batch_size)
            sentences, _ = loader.collate_fn(batch)

            log.info(f'trying batch with {sentences.shape}')

            sentences = sentences.to(self.device)
            self.forward(sentences=sentences)

            res_memory = torch.cuda.memory_reserved(self.device)
            memory_usage = (res_memory / total_memory) * 100
            log.info(
                f'! {loader.dataset.name} is using'
                f' ~{int(memory_usage)}% memory')

            del sentences
            torch.cuda.empty_cache()

        log.info('finished memory check')
        if not test and not trained:
            self.eval()

    #
    # HOOKS
    #

    def on_fit_start(self):
        log.info('running custom pre-training sanity check')
        self.run_memcheck()

    def on_train_epoch_start(self):
        log.info(
            f'! starting epoch {self.current_epoch}'
            f' (step={self.global_step});'
            f' running on {self.device}')

        self.init_projections()

        assert not self.keen.entity_embeddings.weight.requires_grad
        assert not self.keen.relation_embeddings.weight.requires_grad

    def on_train_epoch_end(self, outputs):
        self._has_trained = True

    def on_validation_epoch_start(self):
        # have not figured out to do it any other way
        try:
            self._last_kgc_metrics
        except AttributeError:
            self._last_kgc_metrics = self._mock_kgc_results()

    def on_validation_epoch_end(self):
        """

        Transductive and inductive knowledge graph completion
        with projected entity mappings (see Mapper docstring)

        """
        if any((
            (self.global_step == 0 and not self.trainer.fast_dev_run),
            (not self._has_trained)
        )):
            log.info('skipping kgc evaluation')
            return

        log.info(
            f'[{hvd.local_rank()}] gathered'
            f' {int(self.projections_counts.sum().item())}'
            ' projections')

        self.projections = hvd.allreduce(
            self.projections,
            op=hvd.Sum)

        self.projections_counts = hvd.allreduce(
            self.projections_counts,
            op=hvd.Sum)

        if hvd.local_rank() == 0:
            self._run_kgc_evaluations()

    # ---
