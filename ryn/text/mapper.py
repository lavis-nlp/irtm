# -*- coding: utf-8 -*-

import ryn
from ryn.text import data
from ryn.embers import keen
from ryn.common import helper
from ryn.common import logging

import torch as t
import torch.optim
from torch import nn
import torch.utils.data as torch_data
from torch.nn.utils.rnn import pad_sequence

import transformers as tf
import pytorch_lightning as pl
# from pytorch_lightning.loggers.wandb import WandbLogger as Logger

import gc
import pathlib
import argparse

from functools import partial

from datetime import datetime
from dataclasses import field
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Tuple

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

        return X.max(axis=1).values


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

    def forward(self, X, Y):
        return torch.dist(X, Y, p=2)


_impl(Comparator, EuclideanComparator)


# --- WIRING


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

    # https://github.com/PyTorchLightning/pytorch-lightning/blob/9acee67c31c84dac74cc6169561a483d3b9c9f9d/pytorch_lightning/trainer/trainer.py#L81
    trainer_args: Dict[str, any]

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader_train_args: Dict[str, any]
    dataloader_valid_args: Dict[str, any]

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
    def c(self) -> Components:
        return self._c

    def __init__(self, *, c: Components = None):
        super().__init__()
        self._c = c

        self.encode = c.text_encoder
        self.aggregate = c.aggregator
        self.project = c.projector

        self.loss = c.comparator

    def configure_optimizers(self):
        optim = self.c.Optimizer(self.parameters(), **self.c.optimizer_args)
        log.info(f'initialized optimizer with {self.c.optimizer_args}')
        return optim

    @helper.notnone
    def forward(
            self, *,
            sentences: torch.Tensor = None,  # batch x tokens
            entities: Tuple[int] = None):    # batch

        # mask padding and [MASK] tokens
        mask = self.c.tokenizer.base.vocab['[MASK]']
        attention_mask = (sentences > 0) | (sentences == mask)
        attention_mask = attention_mask.to(dtype=torch.long)

        # batch x tokens x text_dims
        encoded = self.encode(
            input_ids=sentences,
            attention_mask=attention_mask)[0]

        # batch x text_dims
        aggregated = self.aggregate(encoded)

        # batch x kge_dims
        projected = self.project(aggregated)

        # batch x kge_dims
        target = self.c.kgc_model.embeddings(
            entities=entities,
            device=self.device)

        return projected, target

    def training_step(self, batch, batch_idx: int):
        sentences, entities = batch

        projected, target = self.forward(
            sentences=sentences,
            entities=entities)

        loss = self.loss(projected, target)
        return loss

    def validation_step(self, batch, batch_idx: int):
        sentences, entities = batch

        projected, target = self.forward(
            sentences=sentences,
            entities=entities)

        loss = self.loss(projected, target)
        return loss

    # ---

    @classmethod
    def from_config(
            K, *,
            config: MapperConfig = None,
            text_encoder_name: str = None):

        kgc_model = keen.Model.from_path(config.kgc_model)

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


# --- TRAINING


class Dataset(torch_data.Dataset):

    @property
    def token_ids(self) -> Tuple[torch.Tensor]:
        return self._token_ids

    @property
    def entity_ids(self) -> Tuple[int]:
        return self._entity_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return (
            self.token_ids[idx],
            self.entity_ids[idx], )

    def __init__(
            self, *,

            # either
            part: data.Part = None,
            # or
            entity_ids: Tuple[int] = None,
            token_ids: Tuple[torch.Tensor] = None):

        if any((
                not part and not (token_ids and entity_ids),
                part and (token_ids or entity_ids), )):
            assert False, 'provide either a part or token & entity ids'

        if part:
            flat = [
                (e, torch.Tensor(token_ids).to(dtype=torch.long))
                for e, id_lists in part.id2idxs.items()
                for token_ids in id_lists]

            self._entity_ids, self._token_ids = zip(*flat)

        else:
            self._entity_ids = entity_ids
            self._token_ids = token_ids

        assert len(self.entity_ids) == len(self.token_ids)

    def split(self, ratio: float) -> Tuple['Dataset']:
        # must not be shuffled

        n = int(len(self.entity_ids) * ratio)
        e = self.entity_ids[n]

        while self.entity_ids[n] == e:
            n += 1

        log.info(
            f'splitting dataset with param {ratio}'
            f' at {n} ({n / len(self.entity_ids) * 100:2.2f}%)')

        a = Dataset(
            entity_ids=self.entity_ids[:n],
            token_ids=self.token_ids[:n])

        b = Dataset(
            entity_ids=self.entity_ids[n:],
            token_ids=self.token_ids[n:])

        assert not (set(a.entity_ids) & set(b.entity_ids))
        return a, b


def collate_fn(batch: List[Tuple]):
    idxs, ents = zip(*batch)
    return pad_sequence(idxs, batch_first=True), ents


OPTIMIZER = {
    'adam': torch.optim.Adam,
}


def _probe(
        *,
        model: Mapper = None,
        train: torch_data.DataLoader = None,
        valid: torch_data.DataLoader = None, ):

    log.info('probing for functioning configuration')

    max_seq_len = max(
        max(len(seq) for seq, _ in train.dataset),
        max(len(seq) for seq, _ in valid.dataset), )

    log.info(f'determined max sequence length: {max_seq_len}')

    log.info('clean up after probing')

    for p in model.parameters():
        if p.grad is not None:
            del p.grad

    torch.cuda.empty_cache()
    gc.collect()


@helper.notnone
def train(*, config: MapperConfig = None):
    # initializing models
    log.info('initializing models')

    text_dataset = data.Dataset.load(path=config.text_dataset)

    model = Mapper.from_config(
        config=config,
        text_encoder_name=text_dataset.model)

    # # TODO make option
    model.c.text_encoder.eval()
    helper.seed(model.c.kgc_model.ds.cfg.seed)

    assert model.c.tokenizer.base.vocab['[PAD]'] == 0

    # handling data

    ds = Dataset(part=text_dataset.cw_train)
    ds_train, ds_valid = ds.split(config.valid_split)
    # test = Dataset(part=text_dataset.ow_valid)

    DataLoader = partial(torch_data.DataLoader, collate_fn=collate_fn)
    dl_train = DataLoader(ds_train, **config.dataloader_train_args)
    dl_valid = DataLoader(ds_valid, **config.dataloader_valid_args)

    # probe configuration

    _probe(model=model, train=dl_train, valid=dl_valid)

    # torment the machine

    # trainer = pl.Trainer(**config.trainer_args)
    # trainer.fit(model, dl_train, dl_valid)

    log.info('done')


def train_from_args(args: argparse.Namespace):

    DATEFMT = '%Y.%m.%d.%H%M%S'

    # ---

    kgc_model = 'DistMult'
    text_encoder = 'bert-large-cased'
    split_dataset = 'oke.fb15k237_30061990_50'

    kgc_model_dir = f'{kgc_model}-256-2020.08.12.120540.777006'
    text_encoder_dir = f'{text_encoder}.200.768'

    # ---

    now = datetime.now().strftime(DATEFMT)
    name = f'{kgc_model.lower()}.{text_encoder.lower()}.{now}'
    out = ryn.ENV.TEXT_DIR / 'mapper' / split_dataset / name

    out.mkdir(parents=True, exist_ok=True)

    # ---

    logger = pl.loggers.wandb.WandbLogger(
        name=name,
        save_dir=str(out),
        offline=True,
        project='ryn',
        log_model=False,
    )

    # ---

    train(config=MapperConfig(

        # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api
        trainer_args=dict(
            max_epochs=2,
            gpus=1,
            logger=logger,
        ),

        # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        dataloader_train_args=dict(
            num_workers=64,
            batch_size=2,
            shuffle=True,
        ),
        dataloader_valid_args=dict(
            num_workers=64,
            batch_size=2,
        ),

        kgc_model=(
            ryn.ENV.EMBER_DIR / split_dataset / kgc_model_dir),

        text_dataset=(
            ryn.ENV.TEXT_DIR / 'data' / split_dataset / text_encoder_dir),

        optimizer='adam',
        optimizer_args=dict(lr=0.001),

        aggregator='max pooling',
        projector='affine',
        projector_args=dict(input_dims=1024, output_dims=256),
        comparator='euclidean',
        valid_split=0.7,
    ))
