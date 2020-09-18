# -*- coding: utf-8 -*-

import ryn
from ryn.text import data
from ryn.embers import keen
from ryn.graphs import split
from ryn.common import logging

import torch
from torch import nn
from torch.utils import data as tdata

import transformers as tf
import pytorch_lightning as pl

import pathlib
import argparse
from dataclasses import field
from dataclasses import dataclass

from typing import Any
from typing import List
from typing import Dict
from typing import Union

log = logging.get('text.mapper')


class Base(nn.Module):

    @dataclass
    class Config:
        pass

    # to be implemented
    # name = 'name of the module'

    def __init__(self, *args, config: Config = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config

    @classmethod
    def init(K, *, name: str = None, **kwargs):

        A = K.impl[name]
        config = A.Config(**kwargs)

        log.info(f'! initializing {A.__name__} with {config}')
        return A(config=config)


def _impl(Parent, *Klasses):
    Parent.impl = {Klass.name: Klass for Klass in Klasses}


# --- AGGREGATION

class Aggregator(Base):
    pass


class AggregatorMaxPooling(Aggregator):

    name = 'max pooling'

    def forward(
            self,
            X: torch.Tensor  # batch x tokens x text_dims
    ) -> torch.Tensor:       # batch x text_dims

        return X.max(axis=1)


_impl(Aggregator, AggregatorMaxPooling)


# --- PROJECTION


class Projector(Base):
    pass


class AffineProjector(Projector):

    name = 'affine'

    @dataclass
    class Config(Base.Config):

        input_dims: int
        output_dims: int

    # ---

    def __init__(
            self, *args,
            config: 'AffineProjector.Config',
            **kwargs):

        super().__init__(*args, config=config, **kwargs)
        self.projector = nn.Linear(config.input_dims, config.output_dims)

    def forward(
            self,
            X: torch.Tensor  # batch x text_dims
    ) -> torch.Tensor:       # batch x kge_dims
        return self.projector(X)


_impl(Projector, AffineProjector)


# --- COMPARING


class Comparator(Base):
    pass


class EuclideanComparator(Comparator):

    name = 'euclidean'

    def forward(self, batch):
        (
            X,  # batch x kge_dims
            Y   # batch x kge_dims
        ) = batch

        return torch.dist(X, Y, p=2)


_impl(Comparator, EuclideanComparator)


# --- WIRING


@dataclass
class Components:

    # text encoder
    encoder: tf.BertModel

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

    @property
    def c(self) -> Components:
        return self._c

    def __init__(self, *, c: Components = None):
        super().__init__()
        self._c = c

    def forward(self, X):
        print('forward')
        __import__('IPython').embed(); __import__('sys').exit()

    def training_step(self, batch, batch_idx):
        print('training step')
        __import__('IPython').embed(); __import__('sys').exit()
        # Y = self.forward(batch)
        # K =

        # loss = self.comparator(Y, K)

    def validation_step(self, batch, batch_idx):
        print('validation step')
        __import__('IPython').embed(); __import__('sys').exit()


# --- TRAINING


class Dataset(tdata.Dataset):

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    @property
    def entities(self) -> List[int]:
        return self._entities

    def __len__(self):
        return len(self.tokens)

    def __getitem(self, idx: int):
        return self.tokens[idx], self.entities[idx]

    def __init__(self, *, part: data.Part = None):
        flat = [
            (e, toks)
            for e, lis in part.id2toks.items()
            for toks in lis]

        self._entities, self._tokens = zip(*flat)


@dataclass
class Config:

    # ow_valid is split for the training
    # into validation and testing data
    valid_split: int

    # the trained knowledge graph completion model
    # for more information see ryn.embers.keen.Model
    kgc_model: Union[str, pathlib.Path]

    # this is the pre-processed text data
    # and it also determines the upstream text encoder
    # for more information see tyn.text.data.Dataset
    text_dataset: Union[str, pathlib.Path]

    # see the respective <Class>.impl dictionary
    # for available implementations
    # and possibly <Class>.Config for the
    # necessary configuration

    aggregator: str
    projector: str
    comparator: str

    aggregator_args: Dict[str, Any] = field(default_factory=dict)
    projector_args: Dict[str, Any] = field(default_factory=dict)
    comparator_args: Dict[str, Any] = field(default_factory=dict)


def train(*, config=Config):
    # TODO set seed

    # initializing models
    log.info('initializing models')

    kgc_model = keen.Model.from_path(config.kgc_model)

    text_dataset = data.Dataset.load(config.text_dataset)
    text_encoder = tf.BertModel.from_pretrained(
        text_dataset.model,
        cache_dir=ryn.ENV.CACHE_DIR / 'lib.transformers')

    aggregator = Aggregator.init(
        name=config.aggregator,
        **config.aggregator_args)

    projector = Projector.init(
        name=config.projector,
        **config.projector_args)

    comparator = Comparator.init(
        name=config.comparator,
        **config.comparator_args)

    model = Mapper(c=Components(
        encoder=text_encoder,
        aggregator=aggregator,
        projector=projector,
        comparator=comparator,
        kgc_model=kgc_model,
    ))

    # handling data

    train = Dataset(part=text_dataset.cw_train | text_dataset.cw_valid)
    valid, test = tdata.random_split(
        Dataset(part=text_dataset.ow_valid),

        # TODO split needs to be done based on entities not text contexts!
        [int(config.valid_split * len(text_dataset)])

    # torment the machine

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, tdata.DataLoader(train), tdata.DataLoader(valid))


def train_from_args(args: argparse.Namespace):

    split_dataset = 'oke.fb15k237_30061990_50'

    train(config=Config(

        kgc_model=(
            ryn.ENV.EMBER_DIR /
            split_dataset / 'DistMult-256-2020.08.12.120540.777006'),

        text_dataset=(
            ryn.ENV.TEXT_DIR /
            'data' / split_dataset / 'bert-large-cased.200.256'),

        aggregator='max pooling',
        projector='affine',
        projector_args=dict(input_dims=768, output_dims=256),
        comparator='euclidean',
        valid_split=0.7,
    ))
