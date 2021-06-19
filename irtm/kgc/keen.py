# -*- coding: utf-8 -*-

"""

pykeen integration

https://github.com/pykeen/pykeen

"""

import irt

from irtm.kgc import data
from irtm.common import helper
from irtm.kgc.config import Config

import pathlib
import logging
import textwrap

from functools import lru_cache
from dataclasses import dataclass

import torch
import pandas as pd

from pykeen.models import base as keen_models_base
from pykeen.evaluation.rank_based_evaluator import RankBasedMetricResults

from typing import Set
from typing import List
from typing import Union
from typing import Tuple
from typing import Collection


log = logging.getLogger(__name__)
DATEFMT = "%Y.%m.%d.%H%M%S"


# ---


def _triples_to_set(t: torch.Tensor) -> Set[Tuple[int]]:
    return set(map(lambda triple: tuple(triple.tolist()), t))


@dataclass
class Model:
    """

    Trained model

    Use Model.load(...) to load all data from disk.
    The triple factories used are saved within the torch instance.

    """

    config: Config
    path: pathlib.Path

    kcw: irt.KeenClosedWorld
    training_result: data.TrainingResult

    # --

    @property
    def name(self) -> str:
        return self.config.model.cls

    @property
    def dimensions(self) -> int:
        return self.config.model.embedding_dim

    @property
    def uri(self) -> str:
        # assuming the timestamp is unique...
        return (
            f"{self.kcw.dataset.name}/"
            f"{self.name}/"
            f"{self.timestamp.strftime(DATEFMT)}"
        )

    @property
    def keen(self) -> keen_models_base.Model:  # a torch.nn.Module
        return self.training_result.model

    def freeze(self):
        log.info("! freezing kgc model")
        self.keen.requires_grad_(False)

    def unfreeze(self):
        log.info("! un-freezing kgc model")
        self.keen.requires_grad_(True)

    @property
    @lru_cache
    def metrics(self) -> pd.DataFrame:
        metrics = self.test_results["metrics"]
        hits_at_k = metrics["hits_at_k"]["both"]

        data = {}
        for i in (1, 3, 5, 10):
            data[f"hits@{i}"] = {
                kind: hits_at_k[kind][f"{i}"] for kind in ("avg", "best", "worst")
            }

        data["MR"] = metrics["mean_rank"]["both"]
        data["MRR"] = metrics["mean_reciprocal_rank"]["both"]

        return pd.DataFrame(data)

    # ---

    def __str__(self) -> str:
        title = f"\nKGC Model {self.name}-{self.dimensions}\n"

        return title + textwrap.indent(
            f'Trained: {self.timestamp.strftime("%d.%m.%Y %H:%M")}\n'
            f"Dataset: {self.kcw.dataset.name}\n\n"
            f"{self.metrics}\n",
            "  ",
        )

    def __hash__(self) -> int:
        return hash(self.uri)

    #
    # ---  PYKEEN ABSTRACTION
    #

    # translate irtm graph ids to pykeen ids

    def triple2id(self, htr: Tuple[int]) -> Tuple[int]:
        h, t, r = htr

        assert h in self.kcw.dataset.graph.source.ents
        assert t in self.kcw.dataset.graph.source.ents
        assert r in self.kcw.dataset.graph.source.rels

        return self.e2id(h), self.r2id(r), self.e2id(t)

    def triples2id(self, triples: Collection[Tuple[int]]) -> List[Tuple[int]]:
        return list(map(self.triple2id, triples))

    @property
    @lru_cache
    def mapped_train_triples(self) -> Set[Tuple[int]]:
        return _triples_to_set(self.triple_factories.train.mapped_triples)

    @property
    @lru_cache
    def mapped_valid_triples(self) -> Set[Tuple[int]]:
        return _triples_to_set(self.triple_factories.valid.mapped_triples)

    @property
    @lru_cache
    def mapped_test_triples(self) -> Set[Tuple[int]]:
        return _triples_to_set(self.triple_factories.test.mapped_triples)

    def embeddings(self, entities: Collection[int], device: torch.device):
        # mapped = tuple(map(self.e2id, entities))
        indexes = torch.Tensor(entities).to(dtype=torch.long, device=device)
        return self.keen.entity_embeddings(indexes)

    # ---

    @classmethod
    def load(
        K,
        path: Union[str, pathlib.Path],
        dataset: Union[str, irt.Dataset],
        load_model: bool = True,
    ):

        path = helper.path(
            path, exists=True, message="loading keen model from {path_abbrv}"
        )

        config = Config.load(path)
        training_result = data.TrainingResult.load(path)
        dataset = irt.Dataset(dataset) if type(dataset) is str else dataset

        assert (
            config.general.dataset == dataset.name
        ), f"{config.general.dataset=} {dataset.name=}"

        kcw = irt.KeenClosedWorld(
            dataset=dataset,
            seed=config.general.seed or dataset.split.cfg.seed,
            split=config.general.split,
        )

        return K(
            path=path,
            config=config,
            kcw=kcw,
            training_result=training_result,
        )


# copied as unfortunately it is not distributed
# https://github.com/pykeen/pykeen/blob/b33d8c3b13fae5ace983d7b52ea4b9afb72c2cba/tests/mocks.py
def mock_metric_results():

    SIDE_HEAD = "head"
    SIDE_TAIL = "tail"
    SIDE_BOTH = "both"
    SIDES = {SIDE_HEAD, SIDE_TAIL, SIDE_BOTH}

    RANK_OPTIMISTIC = "optimistic"
    RANK_PESSIMISTIC = "pessimistic"
    RANK_REALISTIC = "realistic"
    RANK_TYPES = {RANK_OPTIMISTIC, RANK_PESSIMISTIC, RANK_REALISTIC}

    dummy_1 = {side: {rank_type: 10.0 for rank_type in RANK_TYPES} for side in SIDES}
    dummy_2 = {side: {rank_type: 1.0 for rank_type in RANK_TYPES} for side in SIDES}

    return RankBasedMetricResults(
        arithmetic_mean_rank=dummy_1,
        geometric_mean_rank=dummy_1,
        harmonic_mean_rank=dummy_1,
        median_rank=dummy_1,
        inverse_arithmetic_mean_rank=dummy_2,
        inverse_harmonic_mean_rank=dummy_2,
        inverse_geometric_mean_rank=dummy_2,
        inverse_median_rank=dummy_2,
        adjusted_arithmetic_mean_rank=dummy_2,
        adjusted_arithmetic_mean_rank_index={
            side: {RANK_REALISTIC: 0.0} for side in SIDES
        },
        rank_std=dummy_1,
        rank_var=dummy_1,
        rank_mad=dummy_1,
        hits_at_k={
            side: {rank_type: {10: 0} for rank_type in RANK_TYPES} for side in SIDES
        },
    )
