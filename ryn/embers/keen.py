# -*- coding: utf-8 -*-

"""

pykeen integration

https://github.com/pykeen/pykeen

"""

import ryn
from ryn.graphs import split
from ryn.common import logging

import json
import copy
import random
import pathlib
import textwrap

from datetime import datetime
from dataclasses import dataclass

import torch

import numpy as np
import pandas as pd

from pykeen import pipeline
from pykeen import triples as keen_triples
from pykeen.models import base as keen_base

from typing import Any
from typing import Dict
from typing import Union


log = logging.get('embers.keen')


# ---


DATEFMT = '%Y.%m.%d.%H%M%S.%f'


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
        assert self.valid.num_entities <= self.train.num_entities, (
            f'{self.valid.num_entities=} > {self.train.num_entities=}')
        assert self.test.num_entities <= self.valid.num_entities, (
            f'{self.test.num_entities=} > {self.valid.num_entities=}')

        # all entities must be known at training time
        # (this implicitly checks if there are entities with the same name)
        assert self.train.num_entities == len(self.ds.cw_train.entities), (
            f'{self.train.num_entities=} != {len(self.ds.cw_train.entities)=}')

    @classmethod
    def create(K, ds: split.Dataset) -> 'TripleFactories':

        log.info(f'creating triple factories from {ds.path}')

        def _triple_to_str(htr):
            nonlocal ds
            h, t, r = htr

            return (
                 f'{h}:{ds.g.source.ents[h]}',
                 f'{t}:{ds.g.source.ents[t]}',
                 f'{r}:{ds.g.source.rels[r]}', )

        def _to_ndarray(triples):
            # transform triples to ndarray and re-arrange
            # triple columns from (h, t, r) to (h, r, t)
            return np.array(list(map(_triple_to_str, triples)))[:, (0, 2, 1)]

        log.info(f'setting seed to {ds.cfg.seed}')
        random.seed(ds.cfg.seed)

        # keen uses its own internal indexing
        # so strip own indexes and create "translated" triple matrix
        train, valid = keen_triples.TriplesFactory(
            triples=_to_ndarray(ds.cw_train.triples)
        ).split(
            ds.cfg.train_split, random_state=ds.cfg.seed)

        test = keen_triples.TriplesFactory(
            triples=_to_ndarray(ds.cw_valid.triples),
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

    # is attempted to be loaded but might fail
    dataset: Union[split.Dataset, None]

    results: Dict[str, Any]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

    keen: keen_base.Model  # which is a torch.nn.Module

    @property
    def name(self) -> str:
        return self.parameters['model']

    @property
    def dimensions(self) -> int:
        return self.parameters['model_kwargs']['embedding_dim']

    @property
    def e2id(self) -> Dict[str, int]:
        return self.keen.triples_factory.entity_to_id

    @property
    def r2id(self) -> Dict[str, int]:
        return self.keen.triples_factory.relation_to_id

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
