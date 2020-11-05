# -*- coding: utf-8 -*-

from ryn.common import helper

import json

import pathlib
import dataclasses
from dataclasses import field
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import Union


@dataclass
class Config:

    # whether to fine-tune the text_encoder
    freeze_text_encoder: bool

    # https://github.com/PyTorchLightning/pytorch-lightning/blob/9acee67c31c84dac74cc6169561a483d3b9c9f9d/pytorch_lightning/trainer/trainer.py#L81
    trainer_args: Dict[str, any]

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader_train_args: Dict[str, any]
    dataloader_valid_args: Dict[str, any]

    # ow_valid is split for the training
    # into validation and testing data
    valid_split: int

    # the trained knowledge graph completion model
    # for more information see ryn.kgc.keen.Model
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

    def save(self, path: Union[str, pathlib.Path]):
        fname = 'config.json'
        path = helper.path(
            path, create=True,
            message=f'saving {fname} to {{path_abbrv}}')

        with (path / fname).open(mode='w') as fd:
            json.dump(dataclasses.asdict(self), fd, indent=2)

    @classmethod
    def load(K, path: Union[str, pathlib.Path]) -> 'Config':
        with path.open(mode='r') as fd:
            raw = json.load(fd)

        return K(**raw)
