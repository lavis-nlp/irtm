# -*- coding: utf-8 -*-


from irtm.common import ryaml
from irtm.common import helper
from irtm.kgc.config import Config

import yaml
import torch

import logging
import pathlib
import textwrap
import dataclasses
from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass

from typing import List
from typing import Dict
from typing import Union


log = logging.getLogger(__name__)


def _save_model(path: pathlib.Path = None, model=None):
    fname = "model.ckpt"
    path = helper.path(path, exists=True, message=f"saving {fname} to {{path_abbrv}}")

    torch.save(model, str(path / fname))


def _load_model(path: pathlib.Path = None):
    fname = "model.ckpt"
    path = helper.path(
        path, exists=True, message=f"loading {fname} from {{path_abbrv}}"
    )

    return torch.load(str(path / fname))


# --------------------


@dataclass
class Time:
    @property
    def took(self) -> timedelta:
        return self.end - self.start

    start: datetime
    end: datetime

    @classmethod
    def create(K, dic: Dict[str, str]):
        return K(**{k: datetime.fromisoformat(v) for k, v in dic.items()})


@dataclass
class TrainingResult:

    created: datetime
    git_hash: str
    config: Config

    # metrics
    training_time: Time
    model: torch.nn.Module

    # from pykeen
    results: Dict
    losses: List[float]
    best_metric: float

    # TODO make optional
    wandb: Dict

    @property
    def description(self):

        s = f"training result for {self.config.model.cls}\n"
        s += textwrap.indent(
            f"created: {self.created}\n"
            f"git hash: {self.git_hash}\n"
            f"training took: {self.training_time.took}\n"
            f"dataset: {self.config.general.dataset}\n"
            f"seed: {self.config.general.seed}\n"
            f"best metric: {self.best_metric:.2f}\n"
            "",
            " " * 2,
        )

        return s

    @property
    def result_dict(self):
        return dict(
            created=self.created,
            git_hash=self.git_hash,
            # metrics
            training_time=dataclasses.asdict(self.training_time),
            results=self.results,
            losses=self.losses,
            best_metric=self.best_metric,
            wandb=self.wandb,
        )

    def _save_results(self, path):
        fname = "training.yml"
        path = helper.path(
            path, exists=True, message=f"saving {fname} to {{path_abbrv}}"
        )

        with (path / fname).open(mode="w") as fd:
            yaml.dump(self.result_dict, fd)

    def save(self, path: Union[str, pathlib.Path]):
        path = helper.path(path, create=True)

        self.config.save(path)
        self._save_results(path)
        _save_model(path=path, model=self.model)

        with (path / "summary.txt").open(mode="w") as fd:
            fd.write(self.description)

    @classmethod
    def load(K, path: Union[str, pathlib.Path], load_model: bool = True):
        # TODO instead of load_model: lazy load self.model

        path = helper.path(
            path,
            exists=True,
            message="loading training results from {path_abbrv}",
        )

        raw = ryaml.load(configs=[path / "training.yml"])

        model = None
        if load_model:
            model = torch.load(str(path / "model.ckpt"), map_location="cpu")

        return K(
            **{
                **raw,
                **dict(
                    training_time=Time.create(raw["training_time"]),
                    config=Config.load(path),
                    model=model,
                ),
            }
        )


@dataclass
class EvaluationResult:

    model: str
    created: datetime
    git_hash: str

    # metrics
    evaluation_time: Time
    metrics: Dict

    @staticmethod
    def _fname(prefix):
        return f"evaluation.{prefix}.yml"

    def save(self, path: Union[str, pathlib.Path], prefix: str = None):
        fname = EvaluationResult._fname(prefix)
        path = helper.path(path, message=f"writing {fname} to {{path_abbrv}}")
        with (path / fname).open(mode="w") as fd:
            yaml.dump(dataclasses.asdict(self), fd)

    @classmethod
    def load(K, path: Union[str, pathlib.Path], prefix: str = None):
        path = helper.path(
            path,
            exists=True,
            message="loading evaluation results from {path_abbrv}",
        )

        fname = EvaluationResult._fname(prefix)
        raw = ryaml.load([path / fname])

        return K(
            model=raw["model"],
            created=raw["created"],
            git_hash=raw["git_hash"],
            evaluation_time=Time.create(raw["evaluation_time"]),
            metrics=raw["metrics"],
        )
