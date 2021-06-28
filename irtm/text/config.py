# -*- coding: utf-8 -*-

import irtm
from irtm.common import ryaml
from irtm.common import helper

import yaml

import pathlib
import dataclasses
from dataclasses import field
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from typing import Sequence


class MisconfigurationError(irtm.IRTMError):
    pass


@dataclass
class Config:
    """

    See conf/text/defaults.yml for a detailed explanation

    """

    # TRAINING

    wandb_args: Optional[Dict[str, Any]]
    checkpoint_args: Optional[Dict[str, Any]]

    trainer_args: Dict[str, Any]
    clip_val: float

    optimizer: str
    optimizer_args: Dict[str, Any]

    # UPSTREAM

    kgc_model: Union[str, pathlib.Path]

    # DATA

    mode: str
    dataset: str

    dataloader_train_args: Dict[str, Any]
    dataloader_valid_args: Dict[str, Any]
    dataloader_test_args: Dict[str, Any]

    # MAPPER

    text_encoder: str
    freeze_text_encoder: bool

    aggregator: str
    reductor: str
    projector: str
    comparator: str

    aggregator_args: Dict[str, Any] = field(default_factory=dict)
    reductor_args: Dict[str, Any] = field(default_factory=dict)
    projector_args: Dict[str, Any] = field(default_factory=dict)
    comparator_args: Dict[str, Any] = field(default_factory=dict)

    # OPTIONAL

    sampler: Optional[str] = None
    sampler_args: Optional[Dict[str, Any]] = None

    scheduler: Optional[str] = None
    scheduler_args: Optional[Dict[str, Any]] = None

    early_stopping: bool = False
    early_stopping_args: Optional[Dict[str, Any]] = None

    # SET AUTOMATICALLY

    # directory to save everything to
    out: Union[str, pathlib.Path] = None

    # ---

    def save(self, path: Union[str, pathlib.Path]):
        path = helper.path(path)
        fname = path.name

        path = helper.path(
            path.parent,
            create=True,
            message=f"saving {fname} to {{path_abbrv}}",
        )

        with (path / fname).open(mode="w") as fd:
            yaml.dump(dataclasses.asdict(self), fd)

    @classmethod
    def load(K, path: Union[str, pathlib.Path]) -> "Config":
        return K.create(configs=[path])

    @classmethod
    def create(K, configs: Sequence[Union[str, pathlib.Path]], **kwargs) -> "Config":
        params = ryaml.load(configs=configs, **kwargs)

        def rcheck(obj, fn, trail="config"):
            for k, v in obj.items():
                new_trail = f"{trail}.{k}"

                if type(v) is dict:
                    rcheck(obj=v, fn=fn, trail=new_trail)
                else:
                    fn(new_trail, k, v)

        def checker(trail, k, v):
            if v is None:
                raise irtm.IRTMError(f"{trail} must not be None")

        rcheck(obj=params, fn=checker)
        return K(**params)
