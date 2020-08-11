# -*- coding: utf-8 -*-

"""

pykeen integration

https://github.com/pykeen/pykeen

"""

import ryn
from ryn.graphs import split
from ryn.graphs import graph
from ryn.common import logging

import json
import copy
import pickle
import random
import pathlib
import textwrap

from datetime import datetime
from functools import partial
from functools import lru_cache
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd

from pykeen import pipeline
from pykeen import triples as keen_triples
from pykeen.models import base as keen_base

from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Union
from typing import Tuple
from typing import Collection


log = logging.get('embers.keen')
DATEFMT = '%Y.%m.%d.%H%M%S.%f'


# ---
# might re-introduce the helper.file_cache at some point...


def _cached_predictions(predict_all):
    def _inner(self: split.Dataset, e: int):
        path = ryn.ENV.CACHE_DIR / 'embers.keen'
        path.mkdir(exist_ok=True, parents=True)
        path /= f'{self.uri}.{e}.pkl'

        log.info(f'looking for {path}')
        if path.is_file():
            with path.open(mode='rb') as fd:
                return pickle.load(fd)

        log.info(f'! cache miss for {path}')
        res = predict_all(self, e=e)

        log.info(f'saving to {path}')
        with path.open(mode='wb') as fd:
            pickle.dump(res, fd)

        return res

    return _inner


# ---


# FIXME
# COPIED FROM PYKEEN BRANCH "improve-novelty-computation"
# REMOVE THIS METHOD WHEN PULL REQUEST 51 IS MERGED


TRIPLES_DF_COLUMNS = [
    'head_id', 'head_label',
    'relation_id', 'relation_label',
    'tail_id', 'tail_label']


# self is a triples factory
def tensor_to_df(self, tensor: torch.LongTensor, **kwargs) -> pd.DataFrame:
    """

    Take a tensor of triples and make a pandas dataframe with labels.

    :param tensor: shape: (n, 3)
        The triples, ID-based and in format
        (head_id, relation_id, tail_id).

    :return:
        A dataframe with n rows, and 6 + len(kwargs) columns.
    """
    # Input validation
    additional_columns = set(kwargs.keys())
    forbidden = additional_columns.intersection(TRIPLES_DF_COLUMNS)
    if len(forbidden) > 0:
        raise ValueError(
            f'The key-words for additional arguments must not be in '
            f'{TRIPLES_DF_COLUMNS}, but {forbidden} were '
            f'used.'
        )

    # convert to numpy
    tensor = tensor.cpu().numpy()
    data = dict(zip(['head_id', 'relation_id', 'tail_id'], tensor.T))

    # vectorized label lookup
    entity_id_to_label = np.vectorize(
        {v: k for k, v in self.entity_to_id.items()}.__getitem__)
    relation_id_to_label = np.vectorize(
        {v: k for k, v in self.relation_to_id.items()}.__getitem__)

    for column, id_to_label in dict(
        head=entity_id_to_label,
        relation=relation_id_to_label,
        tail=entity_id_to_label,
    ).items():
        data[f'{column}_label'] = id_to_label(data[f'{column}_id'])

    # Additional columns
    for key, values in kwargs.items():
        # convert PyTorch tensors to numpy
        if torch.is_tensor(values):
            values = values.cpu().numpy()
        data[key] = values

    # convert to dataframe
    rv = pd.DataFrame(data=data)

    # Re-order columns
    columns = TRIPLES_DF_COLUMNS + sorted(
        set(rv.columns).difference(TRIPLES_DF_COLUMNS))

    return rv.loc[:, columns]


# ---


def e2s(g: graph.Graph, e: int):
    return f'{e}:{g.source.ents[e]}'


def r2s(g: graph.Graph, r: int):
    return f'{r}:{g.source.rels[r]}'


def triple_to_str(g: graph.Graph, htr: Tuple[int]):
    """

    Transform a ryn.graphs triple to a pykeen string representation

    Parameters
    ----------

    g: graph.Graph
      ryn graph instance

    htr: Tuple[int]
      ryn graph triple

    """
    h, t, r = htr
    return e2s(g, h), e2s(g, t), r2s(g, r)


def triples_to_ndarray(g: graph.Graph, triples: Collection[Tuple[int]]):
    """

    Transform htr triples to hrt ndarrays of strings

    Parameters
    ----------

    g: graph.Graph
      ryn graph instance

    htr: Collection[Tuple[int]]
      ryn graph triples

    Returns
    -------

    Numpy array of shape [N, 3] containing triples as
    strings of form hrt.

    """

    # transform triples to ndarray and re-arrange
    # triple columns from (h, t, r) to (h, r, t)
    f = partial(triple_to_str, g)
    return np.array(list(map(f, triples)))[:, (0, 2, 1)]


# ---


@dataclass
class TripleFactories:
    """

    Dataset as required by pykeen

    Using the same split.Dataset must always result in
    exactly the same TripleFactories configuration

    """

    ds: split.Dataset

    train: keen_triples.TriplesFactory
    valid: keen_triples.TriplesFactory
    test: keen_triples.TriplesFactory

    def check(self):
        ds = self.ds
        log.info(f'! running self-check for {ds.path.name} TripleFactories')

        assert self.valid.num_entities <= self.train.num_entities, (
            f'{self.valid.num_entities=} > {self.train.num_entities=}')
        assert self.test.num_entities <= self.valid.num_entities, (
            f'{self.test.num_entities=} > {self.valid.num_entities=}')

        # all entities must be known at training time
        # (this implicitly checks if there are entities with the same name)
        assert self.train.num_entities == len(self.ds.cw_train.entities), (
            f'{self.train.num_entities=} != {len(self.ds.cw_train.entities)=}')

        # all known entities and relations are contained in the mappings

        entities = 0, 2
        relations = 1,

        def _triples_to_set(triples, indexes):
            nonlocal ds
            arr = triples_to_ndarray(ds.g, triples)
            return set(arr[:, indexes].flatten())

        for factory in (self.train, self.valid, self.test):

            _mapped = _triples_to_set(ds.cw_train.triples, entities)
            assert len(factory.entity_to_id.keys() - _mapped) == 0, (
                f'{(len(factory.entity_to_id.keys() - _mapped) != 0)=}')

            _mapped = _triples_to_set(ds.cw_train.triples, relations)
            assert len(factory.relation_to_id.keys() - _mapped) == 0, (
                f'{(len(factory.relation_to_id.keys() - _mapped) == 0)=}')

            _mapped = _triples_to_set(ds.cw_valid.triples, entities)
            assert _mapped.issubset(factory.entity_to_id.keys()), (
                f'{_mapped.issubset(factory.entity_to_id.keys())=}')

            _mapped = _triples_to_set(ds.cw_valid.triples, relations)
            assert _mapped.issubset(factory.entity_to_id.keys()), (
                f'{_mapped.issubset(factory.entity_to_id.keys())=}')

    @classmethod
    def create(K, ds: split.Dataset) -> 'TripleFactories':
        log.info(f'creating triple factories from {ds.path}')

        log.info(f'setting seed to {ds.cfg.seed}')
        random.seed(ds.cfg.seed)

        to_a = partial(triples_to_ndarray, ds.g)

        # keen uses its own internal indexing
        # so strip own indexes and create "translated" triple matrix
        train = keen_triples.TriplesFactory(
            triples=to_a(ds.cw_train.triples), )

        # default split is 80/20
        train, valid = train.split(
            ds.cfg.train_split,
            random_state=ds.cfg.seed, )

        # re-use existing entity/relation mappings
        test = keen_triples.TriplesFactory(
            triples=to_a(ds.cw_valid.triples),
            entity_to_id=train.entity_to_id,
            relation_to_id=train.relation_to_id, )

        # ---

        self = K(ds=ds, train=train, valid=valid, test=test)
        self.check()
        return self


@dataclass
class Model:
    """

    Trained model

    Use Model.from_path(...) to load all data from disk.
    The triple factories used are saved within the torch instance.

    """

    path: pathlib.Path
    timestamp: datetime

    # is attempted to be loaded but may fail
    dataset: Union[split.Dataset, None]

    results: Dict[str, Any]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

    keen: keen_base.Model  # which is a torch.nn.Module

    @property
    def name(self) -> str:
        return self.parameters['model']

    @property
    def uri(self) -> str:
        return (
            f'{self.dataset.name}_'
            f'{self.name}_'
            f'{self.timestamp.strftime(DATEFMT)}')

    @property
    def dimensions(self) -> int:
        return self.parameters['model_kwargs']['embedding_dim']

    @property
    def metrics(self) -> pd.DataFrame:
        metrics = self.results['metrics']

        data = {}
        for i in (1, 3, 5, 10):
            data[f'hits@{i}'] = {
                kind: self.results['metrics']['hits_at_k'][kind][f'{i}']
                for kind in ('avg', 'best', 'worst')
            }

        data['MR'] = metrics['mean_rank']
        data['MRR'] = metrics['mean_reciprocal_rank']

        return pd.DataFrame(data)

    def __str__(self) -> str:
        title = f'\nKGC Model {self.name}-{self.dimensions}\n'

        return title + textwrap.indent(
            f'Trained: {self.timestamp.strftime("%d.%m.%Y %H:%M")}\n'
            f'Dataset: {self.metadata["dataset_name"]}\n\n'
            f'{self.metrics}\n', '  ')

    def __hash__(self) -> int:
        return hash(self.uri)

    # ---

    def e2s(self, e: int) -> str:
        return e2s(self.dataset.g, e)

    def r2s(self, r: int) -> str:
        return r2s(self.dataset.g, r)

    def e2id(self, e: int) -> int:
        return self.keen.triples_factory.entity_to_id[self.e2s(e)]

    def r2id(self, r: int) -> int:
        return self.keen.triples_factory.relation_to_id[self.r2s(r)]

    @property
    @lru_cache
    def mapped_train_triples(self) -> Set[Tuple[int]]:
        return set(map(
            lambda t: tuple(t.tolist()),
            self.keen.triples_factory.mapped_triples))

    #
    # ---  PYKEEN ABSTRACTION
    #

    def predict_scores(self, triples: List[Tuple[int]]) -> Tuple[float]:
        """

        Score a triple list with the KGC model

        Parameters
        ----------

        triples: List[Tuple[int]]
          List of triples with the ids provided by ryn.graphs.graph.Graph
          (not the internally used ids of pykeen!)

        Returns
        -------

        Scores in the order of the associated triples

        """
        assert self.dataset, 'no dataset loaded'

        array = triples_to_ndarray(self.dataset.g, triples)
        batch = self.keen.triples_factory.map_triples_to_id(array)
        batch = batch.to(device=self.keen.device)
        scores = self.keen.predict_scores(batch)

        return [float(t) for t in scores]

    def predict_heads(
            self, *,
            t: int = None,
            r: int = None,
            **kwargs) -> pd.DataFrame:
        """

        See keen_base.Model.predict_heads.

        Parameters
        ----------

        t: int
          tail entity id (using graph.Graph indexes)

        r: int
          relation id (using graph.Graph indexes)

        """
        tstr, rstr = e2s(self.dataset.g, t), r2s(self.dataset.g, r)
        return self.keen.predict_heads(rstr, tstr, **kwargs)

    def predict_tails(
            self, *,
            h: int = None,
            r: int = None,
            **kwargs) -> pd.DataFrame:
        """

        See keen_base.Model.predict_tails

        Parameters
        ----------

        h: int
          head entity id (using graph.Graph indexes)

        r: int
          relation id (using graph.Graph indexes)

        """
        hstr, rstr = e2s(self.dataset.g, h), r2s(self.dataset.g, r)
        return self.keen.predict_tails(hstr, rstr, **kwargs)

    # @_cached_predictions
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

        # build dataframe

        n = len(rids) * len(eids)
        res = res.view(n, 3)

        in_train = list(map(
            lambda t: tuple(t.tolist()) in self.mapped_train_triples,
            res))

        _to_df = partial(tensor_to_df, self.keen.triples_factory)
        df = _to_df(res, scores=y.view((n, )), train=in_train, )

        df = df.sort_values(by='scores', ascending=False)

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
    def from_path(K, path: Union[str, pathlib.Path]):
        log.info(f'loading keen model from {path}')
        path = pathlib.Path(path)

        try:
            _, _, created = path.name.split('-')
            timestamp = datetime.strptime(created, DATEFMT)
        except ValueError as exc:
            log.error(f'cannot read {path}')
            raise exc

        md_path = path / 'metadata.json'
        log.info(f'reading metadata from {md_path}')
        with (md_path).open(mode='r') as fd:
            raw = json.load(fd)
            parameters = raw['pipeline']
            metadata = raw['metadata']

        res_path = path / 'results.json'
        log.info(f'reading results from {res_path}')
        with (res_path).open(mode='r') as fd:
            results = json.load(fd)

        ds_path = ryn.ENV.ROOT_DIR / metadata['dataset_path']
        log.info(f'loading dataset from {ds_path}')
        try:
            dataset = split.Dataset.load(ds_path)
        except FileNotFoundError as exc:
            log.error('dataset could not be found')
            dataset = None
            raise exc

        keen_path = str(path / 'trained_model.pkl')
        log.info(f'loading pykeen model from {keen_path}')
        keen_model = torch.load(keen_path)

        return K(
            path=path,
            timestamp=timestamp,
            results=results,
            parameters=parameters,
            metadata=metadata,
            dataset=dataset,
            keen=keen_model,
        )


# ---


def train(tfs: TripleFactories, **kwargs):

    kwargs = {**dict(
        random_seed=tfs.ds.cfg.seed,
    ), **kwargs}

    return pipeline.pipeline(

        training_triples_factory=tfs.train,
        validation_triples_factory=tfs.valid,
        testing_triples_factory=tfs.test,

        metadata=dict(
            metadata=dict(
                dataset_name=tfs.ds.path.name,
                dataset_path=str(tfs.ds.path),
                graph_name=tfs.ds.g.name,
            ),
            pipeline=copy.deepcopy(kwargs),
        ),

        **kwargs)


@dataclass
class Config:

    emb_dim: int
    batch_size: int
    model: str


# def run(exp: config.Config):
def run():
    log.info('✝ running embers.keen')

    path = ryn.ENV.SPLIT_DIR / 'oke.fb15k237_30061990_50/'

    epochs = 3000
    configs = [
        Config(model='DistMult', emb_dim=256, batch_size=512),
    ]

    ds = split.Dataset.load(path)
    tfs = TripleFactories.create(ds)

    for config in configs:
        print(f'\nrunning {config.model}-{config.emb_dim} {ds.path.name}\n')

        kwargs = dict(
            model=config.model,
            model_kwargs=dict(embedding_dim=config.emb_dim),

            optimizer='Adagrad',
            optimizer_kwargs=dict(lr=0.01),

            # loss='CrossEntropyLoss',

            training_kwargs=dict(
                num_epochs=epochs,
                # batch_size=config.batch_size,
            ),
            evaluation_kwargs=dict(
                # batch_size=config.batch_size,
            ),

            stopper='early',
            stopper_kwargs=dict(
                frequency=5, patience=50, delta=0.002),
        )

        res = train(tfs=tfs, **kwargs)

        fname = '-'.join((
            config.model,
            str(config.emb_dim),
            str(datetime.now().strftime(DATEFMT)), ))

        path = ryn.ENV.EMBER_DIR / ds.path.name / fname
        log.info(f'writing results to {path}')

        res.save_to_directory(str(path))


def train_from_args(args):
    run()
    # log.info('running embers.keen training')
    # config.Config.execute(fconf=args.config, fspec=args.spec, callback=run)


# ---


def _cli(args):
    import IPython

    print()
    if not args.path:
        raise ryn.RynError('please provide a --path')

    m = Model.from_path(args.path)
    print(f'{m}')

    banner = '\n'.join((
        '',
        '-' * 20,
        ' RYN KEEN CLIENT',
        '-' * 20,
        '',
        'variables in scope:',
        '    m: Model',
        '',
    ))

    IPython.embed(banner1=banner)
