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

    # must match Config.text_dataset.model
    text_encoder: str

    # whether to fine-tune the text_encoder
    freeze_text_encoder: bool

    # WANDB
    # ----------------------------------------
    # the following arguments are set by default but can be overwritten:
    #   - name: str
    #   - save_dir: str
    #   - offline: bool
    wandb_args: Optional[Dict[str, Any]]

    # LIGHTNING
    # ----------------------------------------

    # https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-class-api
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/9acee67c31c84dac74cc6169561a483d3b9c9f9d/pytorch_lightning/trainer/trainer.py#L81
    trainer_args: Dict[str, Any]

    # https://pytorch-lightning.readthedocs.io/en/stable/generated/pytorch_lightning.callbacks.ModelCheckpoint.html?highlight=ModelCheckpoint
    checkpoint_args: Optional[Dict[str, Any]]

    # PYTORCH
    # ----------------------------------------

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
    sampler: Optional[str]
    sampler_args: Optional[Dict[str, Any]]

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader_train_args: Dict[str, Any]
    dataloader_valid_args: Dict[str, Any]
    dataloader_test_args: Dict[str, Any]

    # pytorch optimizer
    optimizer: str
    optimizer_args: Dict[str, Any]

    # IRTM
    # ----------------------------------------

    # the trained knowledge graph completion model
    # for more information see irtm.kgc.keen.Model
    kgc_model: Union[str, pathlib.Path]

    # this is the pre-processed text data
    # and it also determines the upstream text encoder
    # (irtm.text.data.Dataset)
    text_dataset: Union[str, pathlib.Path]

    # the triple split (irtm.graphs.split.Dataset)
    split_dataset: Union[str, pathlib.Path]

    # see the respective <Class>.impl dictionary
    # for available implementations
    # and possibly <Class>.Config for the
    # necessary configuration

    aggregator: str
    reductor: str
    projector: str
    comparator: str

    aggregator_args: Dict[str, Any] = field(default_factory=dict)
    reductor_args: Dict[str, Any] = field(default_factory=dict)
    projector_args: Dict[str, Any] = field(default_factory=dict)
    comparator_args: Dict[str, Any] = field(default_factory=dict)

    # OPTIONAL
    # ----------------------------------------

    scheduler: Optional[str] = None
    scheduler_args: Optional[Dict[str, Any]] = None

    early_stopping: bool = False
    early_stopping_args: Optional[Dict[str, Any]] = None

    # ow_valid is split for the training
    # into validation and testing data
    valid_split: Optional[int] = None

    # DEFAULTING
    # ----------------------------------------

    # whether to split the text dataset to have geometric
    # inductive/transductive validation steps. Requires
    # each entitiy to have at least two sentences.
    split_text_dataset: bool = True

    # SET AUTOMATICALLY
    # ----------------------------------------

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
    def load(
        K,
        path: Union[str, pathlib.Path],
    ) -> "Config":
        return K.create(configs=[path])

    @classmethod
    def create(
        K,
        configs: Sequence[Union[str, pathlib.Path]],
        **kwargs,
    ) -> "Config":
        configs = [irtm.ENV.CONF_DIR / "text" / "defaults.yml"] + list(configs)
        params = ryaml.load(configs=configs, **kwargs)
        return K(**params)
