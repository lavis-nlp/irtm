# -*- coding: utf-8 -*-

from pykeen import models as pk_models
from pykeen import losses as pk_losses
from pykeen import sampling as pk_sampling
from pykeen import training as pk_training
from pykeen import stoppers as pk_stoppers
from pykeen import trackers as pk_trackers
from pykeen import evaluation as pk_evaluation
from pykeen import optimizers as pk_optimizers

import pathlib
import dataclasses
from dataclasses import dataclass

from typing import Union


# ---
# RYN RELATED


@dataclass
class Data:

    dataset: Union[str, pathlib.Path]


# ---
# PYKEEN RELATED


@dataclass
class Base:

    @property
    def constructor(self):
        raise NotImplementedError()

    @property
    def kwargs(self):
        d = dataclasses.asdict(self)
        del d['cls']
        return d

    cls: str


@dataclass
class Tracker(Base):
    @property
    def constructor(self):
        return pk_trackers.get_result_tracker_cls

    project: str
    experiment: str
    reinit: bool


@dataclass
class Loss(Base):
    @property
    def constructor(self):
        return pk_losses.get_loss_cls


@dataclass
class Model(Base):
    @property
    def constructor(self):
        return pk_models.get_model_cls

    embedding_dim: int
    preferred_device: str
    automatic_memory_optimization: bool = True


@dataclass
class Optimizer(Base):
    @property
    def constructor(self):
        return pk_optimizers.get_optimizer_cls

    lr: int


@dataclass
class Evaluator(Base):
    @property
    def constructor(self):
        return pk_evaluation.get_evaluator_cls

    batch_size: int


@dataclass
class Stopper(Base):
    @property
    def constructor(self):
        return pk_stoppers.get_stopper_cls

    frequency: int
    patience: int
    relative_delta: float


@dataclass
class Sampler(Base):
    @property
    def constructor(self):
        return pk_sampling.get_negative_sampler_cls

    num_negs_per_pos: int


@dataclass
class TrainingLoop(Base):
    @property
    def constructor(self):
        return pk_training.get_training_loop_cls


@dataclass
class Training:
    num_epochs: int
    batch_size: int


@dataclass
class Config:

    # ryn

    data: Data

    # pykeen

    tracker: Tracker
    loss: Loss
    model: Model
    optimizer: Optimizer
    evaluator: Evaluator
    stopper: Stopper
    sampler: Sampler
    training_loop: TrainingLoop
    training: Training

    def resolve(self, option, **kwargs):
        return option.constructor(option.cls)(**{**option.kwargs, **kwargs})
