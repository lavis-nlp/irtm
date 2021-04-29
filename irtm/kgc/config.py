# -*- coding: utf-8 -*-

import irtm
from irtm.common import ryaml
from irtm.common import helper

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
import logging
import pathlib
import dataclasses
from dataclasses import dataclass

import typing
from typing import Any
from typing import Dict
from typing import Union


log = logging.getLogger(__name__)


# ---
# IRTM RELATED


@dataclass
class General:

    dataset: str = None
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

    cls: str
    kwargs: Dict[str, Any]


@dataclass
class Tracker(Base):
    getter = pk_trackers.get_result_tracker_cls


# model


@dataclass
class Model(Base):
    getter = pk_models.get_model_cls


@dataclass
class Optimizer(Base):
    getter = pk_optimizers.get_optimizer_cls


@dataclass
class Regularizer(Base):
    getter = pk_regularizers.get_regularizer_cls


# training


@dataclass
class Loss(Base):
    getter = pk_losses.get_loss_cls


@dataclass
class Evaluator(Base):
    getter = pk_evaluation.get_evaluator_cls


@dataclass
class Stopper(Base):
    getter = pk_stoppers.get_stopper_cls


@dataclass
class Sampler(Base):
    getter = pk_sampling.get_negative_sampler_cls


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

    trials: int
    maximise: bool = False
    study_name: str = None


@dataclass
class Suggestion:
    @staticmethod
    def create(**kwargs):
        # quite arbitrary heuristics in here
        if "low" in kwargs and "high" in kwargs:
            if any(type(kwargs[k]) is float for k in kwargs):
                return FloatSuggestion(**kwargs)
            return IntSuggestion(**kwargs)

        raise irtm.IRTMError(f"cannot create suggestion from {kwargs}")


@dataclass
class FloatSuggestion(Suggestion):

    low: float
    high: float
    step: float = None
    log: bool = False
    initial: float = None

    @helper.notnone
    def suggest(
        self,
        *,
        name: str = None,
        trial: optuna.trial.Trial = None,
    ) -> float:
        return trial.suggest_float(
            name, self.low, self.high, step=self.step, log=self.log
        )


@dataclass
class IntSuggestion(Suggestion):

    low: int
    high: int
    step: int = 1
    log: bool = False
    initial: int = None

    @helper.notnone
    def suggest(
        self,
        *,
        name: str = None,
        trial: optuna.trial.Trial = None,
    ) -> int:
        return trial.suggest_int(
            name, self.low, self.high, step=self.step, log=self.log
        )


# wiring


@dataclass
class Config:

    # irtm

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
            log.error(f"failed to resolve {option.cls} with {option.kwargs}")
            raise exc

    def save(self, path: Union[str, pathlib.Path]):
        fname = "config.json"
        path = helper.path(
            path, create=True, message=f"saving {fname} to {{path_abbrv}}"
        )

        def _wrap(key, val):
            try:
                val = {
                    # e.g. attr=embedding_dim
                    **{
                        attr: (
                            # suggestions
                            dataclasses.asdict(val)
                            if dataclasses.is_dataclass(val)
                            else val
                        )
                        for attr, val in val["kwargs"].items()
                    },
                    **{"cls": val["cls"]},
                }

            # val['kwargs'] and val['cls'] triggers this
            except KeyError:
                pass

            return val

        dic = {
            # e.g. key=model
            key: _wrap(key, val)
            for key, val in dataclasses.asdict(self).items()
        }

        with (path / fname).open(mode="w") as fd:
            json.dump(dic, fd, indent=2)

    @classmethod
    def load(
        K,
        path: Union[str, pathlib.Path],
        fname: str = None,
    ) -> "Config":

        path = helper.path(path)
        path = helper.path(
            path / (fname or "config.json"),
            exists=True,
            message="loading kgc config from {path_abbrv}",
        )

        raw = ryaml.load(configs=[path])

        # there are two levels to consider
        # 1: Config attributes (resolved by type hints)
        # 2: Possible Suggestion instances

        constructors = typing.get_type_hints(K)

        def _unwrap(key, section):
            kwargs = section
            if "cls" in section:
                kwargs = dict(
                    cls=section["cls"],
                    kwargs={
                        # e.g. attr=embedding_dim
                        attr: (
                            Suggestion.create(**val)
                            if type(val) is dict
                            else val
                        )
                        for attr, val in section.items()
                        if attr != "cls"
                    },
                )

            return constructors[key](**kwargs)

        return K(
            **{
                # e.g. key=model
                key: _unwrap(key, section)
                for key, section in raw.items()
            }
        )

    # optuna related

    @property
    def suggestions(self):
        suggestions = {}
        for name, option in self.__dict__.items():
            try:
                for key, val in option.kwargs.items():
                    if isinstance(val, Suggestion):
                        suggestions[f"{name}.{key}"] = val

            # option.kwargs triggers this
            except AttributeError:
                pass

        if not suggestions:
            raise irtm.IRTMError("no parameters marked for optimization")

        return suggestions

    def suggest(self, trial) -> "Config":
        replaced = {}
        # for name, option in self.__dict__.items():
        #     try:
        #         for key, val in option.kwargs.items():
        #             if isin

        #     _dic = {}
        #     for key, val in option.__dict__.items():
        #         if isinstance(val, Suggestion):
        #             _dic[key] = val

        #     suggestions = {
        #         k: v.suggest(name=f'{name}.{k}', trial=trial)
        #         for k, v in _dic.items()
        #     }

        #     if suggestions:
        #         log.info('obtained suggestions: ' + ', '.join(
        #             f'{k}={v}' for k, v in suggestions.items()))
        for name, option in self.__dict__.items():
            try:
                # suggest concrete values
                suggestions = {
                    k: v.suggest(name=f"{name}.{k}", trial=trial)
                    for k, v in option.kwargs.items()
                    if isinstance(v, Suggestion)
                }

                if suggestions:
                    option = self.__dict__[name]
                    kwargs = {**option.kwargs, **suggestions}
                    replaced[name] = dataclasses.replace(option, kwargs=kwargs)

            # triggered by option.kwargs
            except AttributeError:
                continue

        if not replaced:
            raise irtm.IRTMError("no parameters marked for optimization")

        return dataclasses.replace(self, **replaced)
