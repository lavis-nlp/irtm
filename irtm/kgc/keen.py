# -*- coding: utf-8 -*-

"""

pykeen integration

https://github.com/pykeen/pykeen

"""

import irt
from irt.data.pykeen import triples2factory

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

from typing import Set
from typing import List
from typing import Union
from typing import Tuple
from typing import Optional
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
    evaluation_result: Optional[data.EvaluationResult]

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
        mapped = tuple(map(self.e2id, entities))
        indexes = torch.Tensor(mapped).to(dtype=torch.long, device=device)
        return self.keen.entity_embeddings(indexes)

    # ---

    def predict_scores(self, triples: List[Tuple[int]]) -> Tuple[float]:
        """

        Score a triple list with the KGC model

        Parameters
        ----------

        triples: List[Tuple[int]]
          List of triples with the ids provided by irtm.graphs.graph.Graph
          (not the internally used ids of pykeen!)

        Returns
        -------

        Scores in the order of the associated triples

        """
        assert self.kcw, "no dataset loaded"

        factory = triples2factory(triples=triples, idmap=self.kcw.keen2irt)
        batch = factory.to(device=self.keen.device)
        scores = self.keen.predict_scores(batch)

        return [float(t) for t in scores]

    def _predict_all(self, e: int, tails: bool) -> pd.DataFrame:
        # FIXME h=1333 (warlord) is unknown
        # awaiting https://github.com/pykeen/pykeen/pull/51

        e = self.e2id(e)

        def _to_tensor(dic):
            return torch.Tensor(list(dic.values()))

        rids = _to_tensor(self.keen.triples_factory.relation_to_id)
        eids = _to_tensor(self.keen.triples_factory.entity_to_id)

        target = dict(dtype=torch.long, device=self.keen.device)

        # either hr_batch or rt_batch
        batch = torch.zeros((self.keen.num_relations, 2)).to(**target)
        batch[:, 0], batch[:, 1] = (e, rids) if tails else (rids, e)

        if tails:
            scorer = self.keen.predict_scores_all_tails
        else:
            scorer = self.keen.predict_scores_all_heads

        # entities seem to be returned by insertion order in the triple factory
        # https://github.com/pykeen/pykeen/blob/a2ffc2c278bbb0371cc1e056d6b34729a469df54/src/pykeen/models/base.py#L623
        y = scorer(batch).detach()
        assert y.shape == (self.keen.num_relations, self.keen.num_entities)

        res = torch.zeros((len(rids), len(eids), 3), dtype=torch.long)
        res[:, :, 0 if tails else 2] = e

        for i, r in enumerate(rids):
            res[i, :, 2 if tails else 0] = eids
            res[i, :, 1] = r

        n = len(rids) * len(eids)
        res = res.view(n, 3)

        print("containment")

        # check split the triples occur in
        def _is_in(ref):
            return [tuple(triple.tolist()) in ref for triple in res]

        # build dataframe
        # FIXME slow; look at vectorized options (novel in pykeen)
        in_train = _is_in(self.mapped_train_triples)
        in_valid = _is_in(self.mapped_valid_triples)
        in_test = _is_in(self.mapped_test_triples)

        ds = self.kcw.dataset

        cw = ds.cw_train.triples | ds.cw_valid.triples
        in_cw = _is_in(set(self.triples2id(cw)))

        ow = ds.ow_valid.triples | ds.ow_test.triples
        in_ow = _is_in(set(self.triples2id(ow)))

        print("df construction")

        df = self.keen.triples_factory.tensor_to_df(
            res,
            scores=y.view((n,)),
            cw=in_cw,
            ow=in_ow,
            train=in_train,
            valid=in_valid,
            test=in_test,
        )

        df = df.sort_values(by="scores", ascending=False)
        return df

    def predict_all_tails(self, *, h: int = None) -> pd.DataFrame:
        """

        Predict all possible (r, t) for the given h

        Parameters
        ----------

        h: int
          entity id (using graph.Graph indexes)

        """
        assert h is not None
        return self._predict_all(h, True)

    def predict_all_heads(self, *, t: int = None) -> pd.DataFrame:
        """

        Predict all possible (h, r) for the given t

        Parameters
        ----------

        t: int
          entity id (using graph.Graph indexes)

        """
        assert t is not None
        return self._predict_all(t, False)

    # ---

    @classmethod
    def load(
        K,
        path: Union[str, pathlib.Path],
        *,
        dataset: str = None,
        load_model: bool = True,
    ):

        assert dataset

        path = helper.path(
            path, exists=True, message="loading keen model from {path_abbrv}"
        )

        config = Config.load(path)
        training_result = data.TrainingResult.load(path)

        try:
            evaluation_result = data.EvaluationResult.load(path)
        except FileNotFoundError:
            evaluation_result = None

        dataset = irt.Dataset(dataset)
        assert (
            config.general.dataset == dataset.name
        ), f"{config.general.dataset=} {dataset.name=}"

        kcw = irt.KeenClosedWorld(
            dataset=dataset,
            seed=config.general.seed or dataset.split.cfg.seed,
            split=config.general.split,
        )

        # TODO different triple split?
        assert kcw.training.mapped_triples.equal(
            training_result.model.triples_factory.mapped_triples
        ), "cannot reproduce triple split"

        return K(
            path=path,
            config=config,
            kcw=kcw,
            training_result=training_result,
            evaluation_result=evaluation_result,
        )
