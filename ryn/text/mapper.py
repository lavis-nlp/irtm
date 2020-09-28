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
from pytorch_lightning.callbacks.base import Callback
# from pytorch_lightning.loggers.wandb import WandbLogger as Logger

import gc
import yaml
import pathlib
import argparse

from itertools import chain
from itertools import repeat
from functools import partial
from datetime import datetime
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

from typing import Any
from typing import Dict
from typing import List
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
class AggregatorMaxPooling_1(Aggregator):

    name = 'max 1'

    # batch x tokens x text_dims -> batch x text_dims
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.max(axis=1).values


# --- PROJECTION


class Projector(Base):
    pass


@Projector.module
class AffineProjector_1(Projector):
    """

    Y = Ax + b

    """

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
            nn.ReLU(),
            nn.Linear(config.hidden_dims, config.output_dims),
            nn.Tanh(),
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
        sentences, entities = batch

        projected, target = self.forward(
            sentences=sentences,
            entities=entities)

        loss = self.loss(projected, target)

        # --

        result = pl.TrainResult(loss)
        result.log('train_loss_step', loss)

        return result

    #
    #   VALIDATION
    #

    def validation_step(self, batch, batch_idx: int):
        sentences, entities = batch

        projected, target = self.forward(
            sentences=sentences,
            entities=entities)

        loss = self.loss(projected, target)

        # --

        result = pl.EvalResult(loss)
        result.log('valid_loss_step', loss)

        return result

    def validation_epoch_end(self, val_step_outputs: List[pl.EvalResult]):
        avg = val_step_outputs['valid_loss_step'].mean()

        result = pl.EvalResult(avg)
        result.log('valid_loss_epoch', avg)

        return result

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


class TrainerCallback(Callback):

    @property
    def config(self):
        return self._config

    @property
    def dl_train(self) -> torch_data.DataLoader:
        return self._dl_train

    @property
    def dl_valid(self) -> torch_data.DataLoader:
        return self._dl_valid

    def __init__(
            self,
            *args,
            config: MapperConfig = None,
            dl_train: torch_data.DataLoader = None,
            dl_valid: torch_data.DataLoader = None,
            **kwargs):

        super().__init__(*args, **kwargs)
        self._config = config
        self._dl_train = dl_train
        self._dl_valid = dl_valid

    def on_sanity_check_start(self, trainer, mapper):
        log.info('probing for functioning configuration')

        max_seq = []
        for seq, _ in chain(self.dl_train.dataset, self.dl_valid.dataset):
            if len(max_seq) < len(seq):
                max_seq = seq

        log.info(f'determined max sequence length: {len(max_seq)}')

        for batch_size in set((
                self.config.dataloader_train_args['batch_size'],
                self.config.dataloader_valid_args['batch_size'], )):

            log.info(f'testing {batch_size=}')
            sentences = max_seq.repeat(batch_size, 1).to(device=mapper.device)

            mapper(
                sentences=sentences,
                entities=repeat(0, batch_size))

        log.info('clean up after probing')

        for p in mapper.parameters():
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

    # to reproduce runs:
    # pl.seed_everything(model.c.kgc_model.ds.cfg.seed)
    # also pl.Trainer(deterministic=True, ...)

    assert model.c.tokenizer.base.vocab['[PAD]'] == 0

    # handling data

    ds = Dataset(part=text_dataset.cw_train)
    ds_train, ds_valid = ds.split(config.valid_split)
    # test = Dataset(part=text_dataset.ow_valid)

    DataLoader = partial(torch_data.DataLoader, collate_fn=collate_fn)
    dl_train = DataLoader(ds_train, **config.dataloader_train_args)
    dl_valid = DataLoader(ds_valid, **config.dataloader_valid_args)

    # torment the machine

    # callback = TrainerCallback(
    #     config=config,
    #     dl_train=dl_train,
    #     dl_valid=dl_valid)

    trainer = pl.Trainer(**config.trainer_args)
    trainer.fit(model, dl_train, dl_valid)

    log.info('done')


def train_from_args(args: argparse.Namespace):

    DEBUG = True
    if DEBUG:
        log.warning('phony debug run!')

    DATEFMT = '%Y.%m.%d.%H%M%S'

    # ---

    kgc_model = 'DistMult'
    text_encoder = 'bert-base-cased'
    split_dataset = 'oke.fb15k237_30061990_50'

    kgc_model_dir = f'{kgc_model}-256-2020.08.12.120540.777006'
    text_encoder_dir = f'{text_encoder}.200.768-small'

    # ---

    now = datetime.now().strftime(DATEFMT)
    name = f'{kgc_model.lower()}.{text_encoder.lower()}.{now}'
    out = ryn.ENV.TEXT_DIR / 'mapper' / split_dataset / name

    out.mkdir(parents=True, exist_ok=True)

    # ---

    logger = pl.loggers.wandb.WandbLogger(
        name=name,
        save_dir=str(out),
        offline=DEBUG,
        project='ryn',
        log_model=False,
    )

    # ---

    # bert-large-cased: hidden size 1024
    # bert-base-cased: hidden size 768

    train(config=MapperConfig(

        # pytorch lightning trainer
        # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api
        trainer_args=dict(
            max_epochs=2,
            gpus=1,
            logger=logger,
            weights_save_path=out / 'weights',
            # auto_lr_find=True,
            fast_dev_run=DEBUG,
        ),

        # torch dataloader
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        dataloader_train_args=dict(
            num_workers=64,
            batch_size=8,
            shuffle=True,
        ),
        dataloader_valid_args=dict(
            num_workers=64,
            batch_size=8,
        ),

        # ryn upstream
        kgc_model=(
            ryn.ENV.EMBER_DIR / split_dataset / kgc_model_dir),

        text_dataset=(
            ryn.ENV.TEXT_DIR / 'data' / split_dataset / text_encoder_dir),

        # pytorch
        optimizer='adam',
        optimizer_args=dict(lr=0.001),

        # ryn models
        aggregator='max 1',
        projector='mlp 1',
        projector_args=dict(input_dims=768, hidden_dims=500, output_dims=256),
        comparator='euclidean 1',
        valid_split=0.7,
    ))
