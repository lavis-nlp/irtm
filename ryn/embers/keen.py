# -*- coding: utf-8 -*-

"""

pykeen integration

https://github.com/pykeen/pykeen

"""

from ryn.graphs import split
from ryn.common import logging

import random

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
