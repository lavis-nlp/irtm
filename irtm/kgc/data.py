# -*- coding: utf-8 -*-


from irtm.kgc.config import Config
from irtm.common import helper
from irtm.common import logging

import torch

import json
import pathlib
import textwrap
import dataclasses
from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass

from typing import Any
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
    losses: List[float]

    model: torch.nn.Module

    # set by train()
    stopper: Any = None
    result_tracker: Any = None

    # set by Result.load
    wandb: Dict = None

    @property
    def str_stats(self):
        # TODO add metric_results

        s = f"training result for {self.config.model.cls}\n"
        s += textwrap.indent(
            f"created: {self.created}\n"
            f"git hash: {self.git_hash}\n"
            f"training took: {self.training_time.took}\n"
            f"dataset: {self.config.general.dataset}\n"
            f"seed: {self.config.general.seed}\n"
            "",
            " " * 2,
        )

        return s

    @property
    def result_dict(self):
        dic = dict(
            created=self.created,
            git_hash=self.git_hash,
            # metrics
            training_time=dataclasses.asdict(self.training_time),
            losses=self.losses,
            stopper=self.stopper.get_summary_dict(),
        )

        # tracking
        wandb_run = self.result_tracker.run

        dic["wandb"] = dict(
            id=wandb_run.id,
            dir=wandb_run.dir,
            path=wandb_run.path,
            name=wandb_run.name,
            offline=True,
        )

        if not hasattr(wandb_run, "offline"):
            dic["wandb"].update(
                dict(
                    url=wandb_run.url,
                    offline=False,
                )
            )

        return dic

    def _save_results(self, path):
        fname = "training.json"
        path = helper.path(
            path, exists=True, message=f"saving {fname} to {{path_abbrv}}"
        )

        with (path / fname).open(mode="w") as fd:
            json.dump(self.result_dict, fd, default=str, indent=2)

    def save(self, path: Union[str, pathlib.Path]):
        path = helper.path(path, create=True)

        self.config.save(path)
        self._save_results(path)
        _save_model(path=path, model=self.model)

        with (path / "summary.txt").open(mode="w") as fd:
            fd.write(self.str_stats)

    @classmethod
    def load(K, path: Union[str, pathlib.Path], load_model: bool = True):
        # TODO instead of load_model: lazy load self.model

        path = helper.path(
            path,
            exists=True,
            message="loading training results from {path_abbrv}",
        )

        with (path / "training.json").open(mode="r") as fd:
            raw = json.load(fd)

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
        return f"evaluation.{prefix}.json"

    def save(self, path: Union[str, pathlib.Path], prefix: str = None):
        fname = EvaluationResult._fname(prefix)
        path = helper.path(path, message=f"writing {fname} to {{path_abbrv}}")
        with (path / fname).open(mode="w") as fd:
            json.dump(dataclasses.asdict(self), fd, default=str, indent=2)

    @classmethod
    def load(K, path: Union[str, pathlib.Path], prefix: str = None):
        path = helper.path(
            path,
            exists=True,
            message="loading evaluation results from {path_abbrv}",
        )

        fname = EvaluationResult._fname(prefix)
        with (path / fname).open(mode="r") as fd:
            raw = json.load(fd)

        return K(
            model=raw["model"],
            created=raw["created"],
            git_hash=raw["git_hash"],
            evaluation_time=Time.create(raw["evaluation_time"]),
            metrics=raw["metrics"],
        )
