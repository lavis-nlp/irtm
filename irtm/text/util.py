# -*- coding: utf-8 -*-

import irtm
from irtm.common import helper

import csv
import pathlib

from dataclasses import dataclass

from typing import Dict
from typing import Union
from typing import Optional


@dataclass
class Experiment:

    identifier: str
    note: Optional[str]
    machine: Optional[str]
    sentences: int
    mode: str
    exp: str
    agg: str
    frozen: bool
    name: Optional[str]
    run: Optional[str]
    checkpoint: Optional[str]
    split: str
    text: str
    kgc_model: str
    text_model: str
    comment: Optional[str]

    metrics: Optional[Dict] = None

    def __str__(self):
        trail = ".".join((self.identifier, str(self.sentences), self.mode))
        name = f"{self.exp} [{trail}]"

        if self.name:
            name += f" {self.name}"

        return name

    @classmethod
    def from_row(K, row: Dict[str, str]):
        def translate(k: str):
            return {
                "#sents": "sentences",
                "kgc-model": "kgc_model",
                "text-model": "text_model",
                "M": "machine",
            }.get(k, k)

        kwargs = {
            k: v if v else None
            for k, v in ((translate(k), v) for k, v in row.items())
            if k in K.__dataclass_fields__
        }

        kwargs["frozen"] = bool(kwargs["frozen"])
        kwargs["sentences"] = int(kwargs["sentences"])

        return K(**kwargs)

    @property
    def path_checkpoint(self):
        if not all(
            (
                self.name,
                self.run,
                self.checkpoint,
            )
        ):
            raise irtm.IRTMError("missing attributes to create path")

        return helper.path(
            self.path
            / "weights"
            / "irtm-text"
            / self.run
            / "checkpoints"
            / self.checkpoint,
            exists=True,
        )

    @property
    def path(self) -> pathlib.Path:
        if not all(
            (
                self.split,
                self.text,
                self.text_model,
                self.kgc_model,
                self.name,
            )
        ):
            raise irtm.IRTMError("missing attributes to create path")

        return helper.path(
            irtm.ENV.TEXT_DIR
            / "mapper"
            / self.split
            / self.text
            / self.text_model
            / self.kgc_model
            / self.name,
            exists=True,
        )


class Experiments(list):
    @property
    def path(self) -> pathlib.Path:
        return self._path

    def __init__(self, path: Union[str, pathlib.Path]):
        self._path = helper.path(
            path, exists=True, message="loading csv data from {path_abbrv}"
        )

        with self.path.open(mode="r") as fd:
            for row in csv.DictReader(fd):
                if row["identifier"]:
                    self.append(Experiment.from_row(row))


@helper.notnone
def clean(csv_file: str = None):

    exps = util.Experiments(csv_file)
    __import__("pdb").set_trace()
