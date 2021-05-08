# -*- coding: utf-8 -*-

import irtm
from irtm.common import ryaml
from irtm.common import helper

import yaml
import optuna

import logging
import pathlib
import dataclasses
from dataclasses import dataclass

import typing
from typing import Any
from typing import Dict
from typing import Union
from typing import Sequence

log = logging.getLogger(__name__)


# ---
# IRTM RELATED


@dataclass
class General:

    split: Union[float, Sequence[float]]
    dataset: str = None
    seed: int = None


# ---
# PYKEEN RELATED


@dataclass
class Option:

    cls: str
    kwargs: Dict[str, Any]


@dataclass
class Training:

    num_epochs: int


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

    def suggest(
        self,
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

    def suggest(
        self,
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

    tracker: Option

    model: Option
    optimizer: Option
    evaluator: Option
    regularizer: Option

    loss: Option
    stopper: Option

    training: Training
    training_loop: Option

    sampler: Option = None

    # for pipeline.multi
    optuna: Optuna = None

    def save(self, path: Union[str, pathlib.Path]):
        fname = "config.yml"
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
            if val
        }

        with (path / fname).open(mode="w") as fd:
            yaml.dump(dic, fd)

    @classmethod
    def load(
        K,
        path: Union[str, pathlib.Path],
        fname: str = "config.yml",
    ) -> "Config":

        path = helper.path(path)
        path = helper.path(
            path / fname,
            exists=True,
            message="loading kgc config from {path_abbrv}",
        )

        raw = ryaml.load(configs=[path])
        constructors = typing.get_type_hints(K)

        def _unwrap(key, section):
            kwargs = section
            if "cls" in section:
                kwargs = dict(
                    cls=section.get("cls", None),
                    kwargs={
                        # e.g. attr=embedding_dim
                        attr: (Suggestion.create(**val) if type(val) is dict else val)
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
