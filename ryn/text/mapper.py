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
from tqdm import tqdm

# https://github.com/pykeen/pykeen/pull/132
# from pykeen.nn import emb as keen_emb

import yaml

from dataclasses import dataclass
from collections import defaultdict

from typing import Any
from typing import Dict


log = logging.get('text.mapper')


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
class CLSAggregator_1(Aggregator):

    name = 'cls 1'

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[:, 0]


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
        return torch.dist(X, Y, p=2)


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


class Mapper(pl.LightningModule):
    """Maps textual descriptions to knowledge graph embeddings

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
    the internal entity embeddings are overwritten for all owe
    entities and the pykeen evaluator is used to quantify the kgc
    performance of the mapped entity representations.

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
            self, *,
            datasets: data.Datasets = None,
            rync: Components = None):

        super().__init__()
        self._rync = rync
        self._lr = self.rync.optimizer_args['lr']
        self._datasets = datasets

        log.info('freezing kgc model')
        self.rync.kgc_model.keen.eval()

        self.encode = rync.text_encoder
        self.aggregate = rync.aggregator
        self.project = rync.projector

        self.loss = rync.comparator

    def configure_optimizers(self):
        optim = self.rync.Optimizer(
            self.parameters(),
            **self.rync.optimizer_args)

        log.info(f'initialized optimizer with {self.rync.optimizer_args}')
        return optim

    #
    #   SELF REALISATION
    #

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
        assert not self.rync.kgc_model.keen.training
        sentences, entities = batch

        # batch x kge_dims
        projected = self.forward(sentences=sentences)

        # batch x kge_dims
        target = self.forward_entities(entities)

        loss = self.loss(projected, target)
        self.log('train_loss_step', loss)

    #
    #   VALIDATION
    #

    def validation_step(self, batch, batch_idx: int):
        sentences, entities = batch

        # batch x kge_dims
        projected = self.forward(sentences=sentences)

        # batch x kge_dims
        target = self.forward_entities(entities)

        loss = self.loss(projected, target)
        self.log('valid_loss_step', loss)

    def on_validation_epoch_end(self):
        """

        Transductive and inductive knowledge graph completion
        with projected entity mappings (see Mapper docstring)

        """

        # TODO reactivate
        # only happens for fast_dev_run and sanity checks
        # if self.global_step == 0:
        #     # TODO run kgc evaluation as sanity check and for wandb
        #     return

        # TODO replace
        #   kgc_model.entity_embeddings: pykeen.nn.emb.Embdding

        # TODO assert id mappings are equal for cw entities
        # and relations (self.rync.kgc_model.keen_dataset.training)
        log.info('hook called on validation epoch end')

        # TODO overwrite embeddings with projections

        log.info('running transductive evaluation')

        kgc_model = self.rync.kgc_model
        triples = self.datasets.transductive
        tqdm_kwargs = dict(
            total=len(triples.dataloader),
            position=2,
            ncols=80,
            leave=False,
        )

        # project all sentences to an embedding and
        # accumulate these vectors

        accum = defaultdict(lambda: dict(
            count=0,
            vsum=torch.zeros((kgc_model.keen.embedding_dim, )),
        ))

        gen = enumerate(triples.dataloader)
        for batch_idx, batch in tqdm(gen, **tqdm_kwargs):

            sentences, entities = batch
            sentences = sentences.to(device=self.device)

            projected = self.forward(sentences=sentences)
            for e, v in zip(entities, projected):
                accum[e]['count'] += 1
                accum[e]['vsum'] += v.to(device='cpu')

        # create bov representation
        accum = {e: d['vsum'] / d['count'] for e, d in accum.items()}
        log.info(f'replacing {len(accum)} embeddings')

        original_embeddings = kgc_model.keen.entity_embeddings

        # kgc_model.keen.entity_embeddings = \
        #    keen_emb.Embedding.init_with_device(
        #     original_embeddings.num_embeddings,
        #     original_embeddings.embedding_dim,
        #     self.device,
        # )

        kgc_model.keen.entity_embeddings = torch.nn.Embedding(
            original_embeddings.num_embeddings,
            original_embeddings.embedding_dim,
        ).to(device=self.device)

        E = kgc_model.keen.entity_embeddings.weight
        for e, v in accum.items():
            E[triples.ryn2keen[e]] = v

        transductive_result = kgc_trainer.evaluate(
            train_result=kgc_model.training_result,
            mapped_triples=triples.factory.mapped_triples,
            tqdm_kwargs=tqdm_kwargs,
        )

        kgc_model.keen.entity_embeddings = original_embeddings

        log.info('running inductive evaluation')
        # TODO inductive

        __import__("pdb").set_trace()

    # ---

    @classmethod
    @helper.notnone
    def create(
            K, *,
            config: Config = None,
            models: data.Models = None,
            datasets: data.Datasets = None,
    ):
        log.info('creating mapper from config')

        aggregator = Aggregator.init(
            name=config.aggregator,
            **config.aggregator_args)

        projector = Projector.init(
            name=config.projector,
            **config.projector_args)

        comparator = Comparator.init(
            name=config.comparator,
            **config.comparator_args)

        rync = Components(
            Optimizer=OPTIMIZER[config.optimizer],
            optimizer_args=config.optimizer_args,
            text_encoder=models.text_encoder,
            aggregator=aggregator,
            projector=projector,
            comparator=comparator,
            kgc_model=models.kgc_model,
        )

        model = K(
            datasets=datasets,
            rync=rync,
        )

        return model
