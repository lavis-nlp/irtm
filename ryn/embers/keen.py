# -*- coding: utf-8 -*-

"""

pykeen integration

https://github.com/pykeen/pykeen

"""

import ryn
from ryn.graphs import split
from ryn.common import logging

import random
import pathlib

from dataclasses import dataclass

import numpy as np
from pykeen import triples as keen_triples
from pykeen import pipeline as keen_pipeline


log = logging.get('embers.keen')


@dataclass
class TripleFactories:

    ds: split.Dataset
    train: keen_triples.TriplesFactory
    valid: keen_triples.TriplesFactory
    test: keen_triples.TriplesFactory

    @classmethod
    def create(K, ds: split.Dataset) -> 'TripleFactories':

        log.info(f'creating triple factories from {ds.path}')

        def _triple_to_str(htr):
            nonlocal ds
            h, t, r = htr

            return (
                ds.g.source.ents[h],
                ds.g.source.ents[t],
                ds.g.source.rels[r], )

        def _to_a(triples):
            # transform triples to ndarray and re-arrange
            # triple columns from (h, t, r) to (h, r, t)
            return np.array(list(map(_triple_to_str, triples)))[:, (0, 2, 1)]

        log.info(f'setting seed to {ds.cfg.seed}')
        random.seed(ds.cfg.seed)

        # keen uses its own internal indexing
        # so strip own indexes and create "translated" triple matrix
        train, valid = keen_triples.TriplesFactory(
            triples=_to_a(ds.cw_train.triples)
        ).split(
            ds.cfg.train_split, random_state=ds.cfg.seed)

        test = keen_triples.TriplesFactory(
            triples=_to_a(ds.cw_valid.triples),
            entity_to_id=train.entity_to_id,
            relation_to_id=train.relation_to_id, )

        return K(ds=ds, train=train, valid=valid, test=test)


def train(tfs: TripleFactories, **kwargs):
    return keen_pipeline.pipeline(
        random_seed=tfs.ds.cfg.seed,
        training_triples_factory=tfs.train,
        validation_triples_factory=tfs.valid,
        testing_triples_factory=tfs.test,
        **kwargs)


@dataclass
class Config:

    emb_dim: int
    batch_size: int
    model: str


# def run(exp: config.Config):
def run():
    log.info('✝ running embers.keen')

    path = pathlib.Path('data/split/oke.fb15k237_30061990_50/')

    epochs = 2000
    configs = [
        # batch_size currently unused
        Config(model='DistMult', emb_dim=256, batch_size=320),
    ]

    ds = split.Dataset.load(path)
    tfs = TripleFactories.create(ds)

    for config in configs:
        print(f'\nrunning {config.model}-{config.emb_dim}\n')

        kwargs = dict(
            model=config.model,
            model_kwargs=dict(embedding_dim=config.emb_dim),

            training_kwargs=dict(
                num_epochs=epochs,
                # batch_size=config.batch_size,
            ),

            evaluation_kwargs=dict(
                # batch_size=config.batch_size,
            ),

            stopper='early',
            stopper_kwargs=dict(
                frequency=5, patience=50, delta=0.0002),
        )

        res = train(tfs=tfs, **kwargs)

        path = ryn.ENV.EMBER_DIR / f'{kwargs["model"]}-{config.emb_dim}'
        log.info(f'writing results to {path}')

        res.save_to_directory(str(path))


def train_from_args(args):
    run()
    # log.info('running embers.keen training')
    # config.Config.execute(fconf=args.config, fspec=args.spec, callback=run)
