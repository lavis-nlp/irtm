# -*- coding: utf-8 -*-

import ryn
from ryn.common import helper
from ryn.common import logging

import optuna

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
import dataclasses
from dataclasses import dataclass

from typing import Any
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

    getter = None

    @property
    def constructor(self):
        """
        Returns the class defined by self.cls
        """
        return self.__class__.getter(self.cls)

    @property
    def kwargs(self):
        d = dataclasses.asdict(self)
        del d['cls']
        return d

    cls: str


@dataclass
class Tracker(Base):

    getter = pk_trackers.get_result_tracker_cls

    project: str
    experiment: str

    reinit: bool = False
    offline: bool = False


# model


@dataclass
class Model(Base):

    getter = pk_models.get_model_cls

    embedding_dim: int
    preferred_device: str = 'cuda'  # or cpu
    automatic_memory_optimization: bool = True


@dataclass
class Optimizer(Base):

    getter = pk_optimizers.get_optimizer_cls

    lr: int


@dataclass
class Regularizer(Base):

    getter = pk_regularizers.get_regularizer_cls

    p: float
    weight: float
    normalize: bool


# training


@dataclass
class Loss(Base):

    getter = pk_losses.get_loss_cls

    margin: float
    reduction: str


@dataclass
class Evaluator(Base):

    getter = pk_evaluation.get_evaluator_cls

    # batch_size: int


@dataclass
class Stopper(Base):

    getter = pk_stoppers.get_stopper_cls

    frequency: int
    patience: int
    relative_delta: float


@dataclass
class Sampler(Base):

    getter = pk_sampling.get_negative_sampler_cls

    num_negs_per_pos: int


@dataclass
class TrainingLoop(Base):

    getter = pk_training.get_training_loop_cls


@dataclass
class Training:

    num_epochs: int
    batch_size: int = None


# ---
# OPTUNA RELATED


@dataclass
class Optuna:

    study_name: str
    trials: int
    maximise: bool = False


@dataclass
class Suggestion:

    pass


@dataclass
class FloatSuggestion(Suggestion):

    low:     float
    high:    float
    step:    float = None
    log:      bool = False
    initial: float = None

    @helper.notnone
    def suggest(
            self, *,
            name: str = None,
            trial: optuna.trial.Trial = None,
    ) -> float:
        return trial.suggest_float(
            name, self.low, self.high,
            step=self.step, log=self.log)


@dataclass
class IntSuggestion(Suggestion):

    low:     int
    high:    int
    step:    int = 1
    log:    bool = False
    initial: int = None

    @helper.notnone
    def suggest(
            self, *,
            name: str = None,
            trial: optuna.trial.Trial = None,
    ) -> int:
        return trial.suggest_int(
            name, self.low, self.high,
            step=self.step, log=self.log)


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

    # for pipeline.multi
    optuna: Optuna = None

    def resolve(self, option, **kwargs):
        try:
            getter = option.__class__.getter
            return getter(option.cls)(**{**option.kwargs, **kwargs})
        except TypeError as exc:
            log.error(f'failed to resolve {option.cls} with {option.kwargs}')
            raise exc

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

    # optuna related

    @property
    def suggestions(self):
        suggestions = {}
        for name, option in self.__dict__.items():
            for key, val in option.__dict__.items():
                if isinstance(val, Suggestion):
                    suggestions[f'{name}.{key}'] = val

        if not suggestions:
            raise ryn.RynError('no parameters marked for optimization')

        return suggestions

    def suggest(self, trial) -> 'Config':

        replaced = {}
        for name, option in self.__dict__.items():

            _dic = {}
            for key, val in option.__dict__.items():
                if isinstance(val, Suggestion):
                    _dic[key] = val

            suggestions = {
                k: v.suggest(name=f'{name}.{k}', trial=trial)
                for k, v in _dic.items()
            }

            if suggestions:
                log.info('obtained suggestions: ' + ', '.join(
                    f'{k}={v}' for k, v in suggestions.items()))

            # create a new dataclass instance with the respective
            # fields replaced with the concrete optuna suggestion
            replaced[name] = dataclasses.replace(option, **suggestions)

        if not replaced:
            raise ryn.RynError('no parameters marked for optimization')

        return dataclasses.replace(self, **replaced)
