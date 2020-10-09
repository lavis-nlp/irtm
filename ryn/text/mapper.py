# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.text import data
from ryn.common import helper
from ryn.common import logging

import torch as t
import torch.optim
from torch import nn

import transformers as tf
import pytorch_lightning as pl

import yaml
import pathlib

from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

from typing import Any
from typing import Dict
from typing import Union
from typing import Tuple

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
    tokenizer: data.Tokenizer

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


@dataclass
class MapperConfig:

    # whether to fine-tune the text_encoder
    freeze_text_encoder: bool

    # https://github.com/PyTorchLightning/pytorch-lightning/blob/9acee67c31c84dac74cc6169561a483d3b9c9f9d/pytorch_lightning/trainer/trainer.py#L81
    trainer_args: Dict[str, any]

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader_train_args: Dict[str, any]
    dataloader_valid_args: Dict[str, any]

    # ow_valid is split for the training
    # into validation and testing data
    valid_split: int

    # the trained knowledge graph completion model
    # for more information see ryn.kgc.keen.Model
    kgc_model: Union[str, pathlib.Path]

    # this is the pre-processed text data
    # and it also determines the upstream text encoder
    # for more information see tyn.text.data.Dataset
    text_dataset: Union[str, pathlib.Path]

    optimizer: str
    optimizer_args: Dict[str, Any]

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


class Mapper(pl.LightningModule):

    @property
    def lr(self):
        return self._lr

    @lr.setter  # required for auto_lr_find
    def lr(self, val):
        self._lr = val

    @property
    def c(self) -> Components:
        return self._c

    def __init__(self, *, c: Components = None):
        super().__init__()
        self._c = c
        self._lr = self.c.optimizer_args['lr']

        log.info('setting kgc model to eval')
        self.c.kgc_model.keen.eval()

        self.encode = c.text_encoder
        self.aggregate = c.aggregator
        self.project = c.projector

        self.loss = c.comparator

    def configure_optimizers(self):
        optim = self.c.Optimizer(self.parameters(), **self.c.optimizer_args)
        log.info(f'initialized optimizer with {self.c.optimizer_args}')
        return optim

    #
    #   SELF REALISATION
    #

    def forward_sentences(self, sentences: torch.Tensor):
        # mask padding and [MASK] tokens
        mask = self.c.tokenizer.base.vocab['[MASK]']
        attention_mask = (sentences > 0) | (sentences == mask)
        attention_mask = attention_mask.to(dtype=torch.long)

        return self.encode(
            input_ids=sentences,
            attention_mask=attention_mask)[0]

    def forward_entities(self, entities: torch.Tensor):
        # TODO: add embeddings to self; take care of id mapping
        return self.c.kgc_model.embeddings(
            entities=entities,
            device=self.device)

    @helper.notnone
    def forward(
            self, *,
            sentences: torch.Tensor = None,  # batch x tokens
            entities: Tuple[int] = None):    # batch

        # batch x tokens x text_dims
        encoded = self.forward_sentences(sentences)

        # batch x text_dims
        aggregated = self.aggregate(encoded)

        # batch x kge_dims
        projected = self.project(aggregated)

        # batch x kge_dims
        target = self.forward_entities(entities)

        return projected, target

    #
    #   TRAINING
    #

    def training_step(self, batch, batch_idx: int):
        assert not self.c.kgc_model.keen.training

        sentences, entities = batch

        projected, target = self.forward(
            sentences=sentences,
            entities=entities)

        loss = self.loss(projected, target)
        self.log('train_loss_step', loss)

        return loss

    #
    #   VALIDATION
    #

    def validation_step(self, batchd, batch_idx: int):
        """

        Each validation step makes two passes through the model
        both for the inductive and the transductive setting.

        """
        losses = {}
        for name, batch in batchd.items():
            sentences, entities = batch

            projected, target = self.forward(
                sentences=sentences,
                entities=entities)

            # TODO (after keen training): there is a problem
            # that 'loss' is a zero-value tensor

            losses[name] = loss = self.loss(projected, target)
            self.log(f'valid_loss_{name}_step', loss)

        log.error(f'{tuple(losses.values())=}')
        return torch.cat(tuple(losses.values())).mean()

    # ---

    @classmethod
    def from_config(
            K, *,
            config: MapperConfig = None,
            text_encoder_name: str = None):

        kgc_model = keen.Model.load(config.kgc_model)

        text_encoder = tf.BertModel.from_pretrained(
            text_encoder_name,
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

        model = K(c=Components(
            Optimizer=OPTIMIZER[config.optimizer],
            optimizer_args=config.optimizer_args,
            text_encoder=text_encoder,
            tokenizer=data.Tokenizer(model=text_encoder_name),
            aggregator=aggregator,
            projector=projector,
            comparator=comparator,
            kgc_model=kgc_model,
        ))

        return model
