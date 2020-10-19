# -*- coding: utf-8 -*-

from ryn.common import helper
from ryn.common import logging

from pykeen import models as pk_models
from pykeen import losses as pk_losses
from pykeen import sampling as pk_sampling
from pykeen import training as pk_training
from pykeen import stoppers as pk_stoppers
from pykeen import trackers as pk_trackers
from pykeen import evaluation as pk_evaluation
from pykeen import optimizers as pk_optimizers
from pykeen import regularizers as pk_regularizers

import json
import pathlib
import datetime
import dataclasses
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import Union


log = logging.get('kgc.config')


# ---
# RYN RELATED


@dataclass
class General:

    dataset: str
    seed: int = None


# ---
# PYKEEN RELATED


@dataclass
class Base:

    @property
    def getter(self):
        """
        Returns the function that allows the getter lookup
        """
        raise NotImplementedError()

    @property
    def constructor(self):
        """
        Returns the class defined by self.cls
        """
        return self.getter(self.cls)

    @property
    def kwargs(self):
        d = dataclasses.asdict(self)
        del d['cls']
        return d

    cls: str


@dataclass
class Tracker(Base):
    @property
    def getter(self):
        return pk_trackers.get_result_tracker_cls

    project: str
    experiment: str

    reinit: bool = False
    offline: bool = False


# model


@dataclass
class Model(Base):
    @property
    def getter(self):
        return pk_models.get_model_cls

    embedding_dim: int
    preferred_device: str = 'cuda'  # or cpu
    automatic_memory_optimization: bool = True


@dataclass
class Optimizer(Base):
    @property
    def getter(self):
        return pk_optimizers.get_optimizer_cls

    lr: int


@dataclass
class Regularizer(Base):
    @property
    def getter(self):
        return pk_regularizers.get_regularizer_cls

    p: float
    weight: float
    normalize: bool


# training


@dataclass
class Loss(Base):
    @property
    def getter(self):
        return pk_losses.get_loss_cls

    margin: float
    reduction: str


@dataclass
class Evaluator(Base):
    @property
    def getter(self):
        return pk_evaluation.get_evaluator_cls

    batch_size: int


@dataclass
class Stopper(Base):
    @property
    def getter(self):
        return pk_stoppers.get_stopper_cls

    frequency: int
    patience: int
    relative_delta: float


@dataclass
class Sampler(Base):
    @property
    def getter(self):
        return pk_sampling.get_negative_sampler_cls

    num_negs_per_pos: int


@dataclass
class TrainingLoop(Base):
    @property
    def getter(self):
        return pk_training.get_training_loop_cls


@dataclass
class Training:
    num_epochs: int
    batch_size: int


# wiring


@dataclass
class Config:

    # ryn

    general: General

    # pykeen

    tracker: Tracker

    model: Model
    optimizer: Optimizer
    evaluator: Evaluator
    regularizer: Regularizer

    loss: Loss
    stopper: Stopper
    sampler: Sampler
    training: Training
    training_loop: TrainingLoop

    def resolve(self, option, **kwargs):
        return option.getter(option.cls)(**{**option.kwargs, **kwargs})

    def save(self, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        _path_abbrv = f'{path.parent.name}/{path.name}'
        fname = 'config.json'
        log.info(f'saving config to {_path_abbrv}/{fname}')
        with (path / fname).open(mode='w') as fd:
            json.dump(dataclasses.asdict(self), fd, indent=2)

    @staticmethod
    def load(path: Union[str, pathlib.Path]) -> 'Config':
        raise NotImplementedError()
