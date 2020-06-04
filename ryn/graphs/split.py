# -*- coding: utf-8 -*-

"""

Create graph splits exposing tbox proxies.

"""

import ryn
from ryn.graphs import graph
from ryn.common import logging

import random
import pathlib
import operator

from functools import partial
from itertools import product
from itertools import combinations

from dataclasses import field
from dataclasses import dataclass

import numpy as np
from tabulate import tabulate

from typing import Any
from typing import Set
from typing import List
from typing import Tuple


log = logging.get('graph.split')


def _partition(data: Set[Any], line: int) -> Tuple[Set[Any], Set[Any]]:
    lis = list(data)
    random.shuffle(lis)
    return set(lis[:line]), set(lis[line:])


@dataclass
class Config:

    # split ratio (for example: retaining 70% of all
    # samples for training requires a value of .7)
    owcw_split: float
    trainvalid_split: float

    # no of relation (sorted by ratio)
    threshold: int


@dataclass
class Relation:

    r: int
    name: str
    triples: Set[Tuple[int]]

    hs: Set[int]
    ts: Set[int]

    ratio: float

    @classmethod
    def from_graph(K, g: graph.Graph) -> List['Relation']:
        rels = []
        for r, relname in g.source.rels.items():
            triples = g.find(edges={r})
            hs, ts = map(set, zip(*((h, t) for h, t, _ in triples)))

            lens = len(hs), len(ts)
            ratio = min(lens) / max(lens)

            rels.append(K(
                r=r, name=relname, triples=triples,
                hs=hs, ts=ts,
                ratio=ratio, ))

        return rels


@dataclass
class Split:
    """
    Abstraction for two disjunct sets of things.
    Used for both entity and triple sets.

    """

    a: Set[Any] = field(default_factory=set)
    b: Set[Any] = field(default_factory=set)

    def _apply(self, op, other):
        try:
            return Split(
                train=op(self.a, other.a),
                valid=op(self.b, other.b), )
        except AttributeError:
            return Split(
                train=op(self.a, other),
                valid=op(self.b, other), )

    @property
    def lens(self):
        return len(self.a), len(self.b)

    @property
    def unionized(self):
        return self.a | self.b

    def check(self):
        union = self.a & self.b
        assert len(union) == 0, str(union)

    def intersection(self, other):
        return self._apply(operator.and_, other)

    def union(self, other):
        return self._apply(operator.or_, other)

    def without(self, other):
        return self._apply(operator.sub, other)

    def clone(self):
        return self._apply(operator.and_, self)

    def reorder(self, other: 'Split'):
        train = self.a
        valid = self.b

        trainvalid = self.a & other.b
        train -= trainvalid
        valid |= trainvalid

        validtrain = self.b & other.a
        valid -= validtrain
        train |= validtrain

        split = Split(train=train, valid=valid)
        split.check()

        return split, len(trainvalid | validtrain)

    def partition(self, ratio: float) -> Tuple['Split', 'Split']:
        t1, t2 = _partition(self.a, int(len(self.a) * ratio))
        v1, v2 = _partition(self.b, int(len(self.b) * ratio))

        return Split(a=t1, b=t2), Split(a=v1, b=v2)


@dataclass
class Stats:
    """
    Represents a row of the statistics gathered per dataset
    """

    name: str
    seed: int = -1
    threshold: int = -1

    # total amount of candidate triples
    total: int = -1

    # amount of entities retained as
    # concepts based on the config threshold
    retained: int = -1

    # amount of triples selected based on
    # the retained concept entities
    selected: int = -1

    # remaining triples after first selection
    remaining: int = -1

    # triples that have at least one concept entity
    candidates: int = -1

    # ow/cw split candidates
    cw_candidates: int = -1
    ow_candidates: int = -1
    cwow_remaining: int = -1
    cwow_percent: int = -1

    # final dataset
    cwow_final_percent: int = -1

    cw_final_train: int = -1
    cw_final_valid: int = -1
    ow_final_train: int = -1
    ow_final_test:  int = -1

    forgotten: int = -1

    HEADERS = ('name', 'seed', 'threshold', 'total', 'retained', 'selected',
               'remaining', 'candidates', 'cw_candidates', 'ow_candidates',
               'cwow_remaining', 'cwow_percent', 'cwow_final_percent',
               'cw_final_train', 'cw_final_valid', 'ow_final_train',
               'ow_final_test:', 'forgotten', )

    @property
    def tup(self):
        return (self.name, self.seed, self.threshold, self.total,
                self.retained, self.selected, self.remaining,
                self.candidates, self.cw_candidates,
                self.ow_candidates, self.cwow_remaining,
                self.cwow_percent, self.cwow_final_percent,
                self.cw_final_train, self.cw_final_valid,
                self.ow_final_train, self.ow_final_test,
                self.forgotten, )


def _ents_from_triples(ents):
    hs, ts, _ = zip(*list(ents))
    return set(hs) | set(ts)


def _track(s, x, name=''):
    if x in s.a:
        print(f'{x} found in {name} train')

    elif x in s.b:
        print(f'{x} found in {name} valid')

    else:
        print(f'{x} found in neither train or valid')


# ---


def check(rels, cw, ow, triples, forgotten):

    for s1, s2 in combinations((cw.a, cw.b, ow.a, ow.b), 2):
        assert not s1 & s2, f'{len(s1)=} {len(s2)=} {len(s1 & s2)=}'

    # a, b = len(triples.unionized), len(forgotten)
    # assert a == b, f'unionized: {a} / original: {b}'


# datasets are saved in OpenKE format
def write(
        path: pathlib.Path,
        g: graph.Graph,
        cw: Split, ow: Split,
        forgotten: Set[Tuple[int]]):

    def _write(name, tups):
        with (path / name).open(mode='w') as fd:
            fd.write(f'{len(tups)}\n')
            lines = (' '.join(map(str, t)) for t in tups)
            fd.write('\n'.join(lines))
            log.info(f'wrote {fd.name}')

    _write('cw.train2id.txt', cw.a)
    _write('cw.valid2id.txt', cw.b)
    _write('ow.train2id.txt', ow.a)
    _write('ow.test2id.txt', ow.b)
    _write('forgotten.txt', forgotten)

    _write('relation2id.txt', [(v, k) for k, v in g.source.rels.items()])
    _write('entity2id.txt', [(v, k) for k, v in g.source.ents.items()])


# ---


def _write_tsv(table, path, name):
    with (path / name).open(mode='w') as fd:
        fd.write(table(tablefmt='tsv'))
        log.info(f'wrote {fd.name}')


def stats_rows(path, rows):
    table = partial(tabulate, [r.tup for r in rows], headers=Stats.HEADERS)
    _write_tsv(table, path, 'stats.tsv')


def stats_entity_intersections(path, cw, ow):
    ents = (
        _ents_from_triples(cw.a),
        _ents_from_triples(cw.b),
        _ents_from_triples(ow.a),
        _ents_from_triples(ow.b), )

    names = 'cw.train', 'cw.valid', 'ow.train', 'ow.test'

    intersections = [len(a & b) for a, b in product(ents, ents)]
    intersections = np.array(intersections).reshape((4, 4))

    rows = [
        (names[i], ) + tuple(intersections[i])
        for i in range(len(intersections))]

    table = partial(tabulate, rows, headers=names)
    _write_tsv(table, path, 'stats.ents.tsv')


# ------------------------------------------------------------


# method is used if the triple has to be put in either cw or ow
# because the entity is encountered in one of the splits
def _constrained_add(cwow: Split, entities: Split, triple):
    h, t, r = triple

    if h in entities.a and t in entities.a:
        cwow.a.add(triple)
        return None

    # select unencountered entity
    x = h if t in entities.a else t

    # if already in valid, add triple to ow
    if x in entities.b:
        cwow.b.add(triple)
        return None

    return x


def _incremental_add(cfg: Config, cwow: Split, entities: Split, lis):
    while len(lis):
        triple = lis.pop()
        x = _constrained_add(cwow, entities, triple)
        if not x:
            continue

        # track whether the balance of ow/cw fits the configuration
        ratio = len(cwow.a) / (len(cwow.a) + len(cwow.b))
        if ratio < cfg.owcw_split:
            assert x not in entities.b, x
            entities.a.add(x)
            cwow.a.add(triple)

        else:
            assert x not in entities.a, x
            entities.b.add(x)
            cwow.b.add(triple)


def _create(
        g: graph.Graph,
        cfg: Config,
        rels: List[Relation],
        name: str):

    stats = Stats(name=name)
    stats.threshold = cfg.threshold

    selection = rels[:cfg.threshold]
    remaining = set(g.source.triples.copy())  # triples
    retained = set()  # entities

    for rel in selection:
        # determine whether the range or domain of
        # the relations are assumed to be concepts
        reverse = len(rel.hs) <= len(rel.ts)
        concepts, objects = (rel.hs, rel.ts) if reverse else (rel.ts, rel.hs)
        retained |= concepts

    stats.total = len(remaining)
    stats.retained = len(retained)
    log.info(f'found {len(retained)} concepts ({len(remaining)} total)')

    # triples where both head and tail are
    # retained must stay in the cw set
    cwow = Split()
    cwow.a = g.select(heads=retained, tails=retained)
    remaining -= cwow.a

    stats.selected = len(cwow.a)
    stats.remaining = len(remaining)
    log.info(f'moving {len(cwow.a)} triples to cw'
             f' ({len(remaining)} remain)')

    # find triples that are linked to the concepts
    candidates = list(g.find(heads=retained, tails=retained) & remaining)
    random.shuffle(candidates)

    stats.candidates = len(candidates)
    log.info(f'found {len(candidates)} triples linking concepts')

    # keep track of the entities such that no entities ends up in both cw/ow
    entities = Split(a=retained)
    _incremental_add(cfg, cwow, entities, candidates)
    remaining -= cwow.unionized

    # check constraints
    assert not len(candidates)
    entities.check()

    _p = len(cwow.a) / (len(cwow.a) + len(cwow.b))
    stats.cw_candidates = len(cwow.a)
    stats.ow_candidates = len(cwow.b)
    stats.cwow_remaining = len(remaining)
    stats.cwow_percent = _p
    log.info(f'gathered {len(cwow.a)} cw and '
             f'{len(cwow.b)} ow triples: '
             f'{int(_p) * 100}% cw ({len(remaining)} remaining)')

    # distribute left-over triples
    _incremental_add(cfg, cwow, entities, remaining)

    assert not len(remaining)
    entities.check()
    cwow.check()

    # partition closed/open world triple sets
    # into train/valid (ow) and train/test (cw)
    cw, ow = cwow.partition(cfg.trainvalid_split)

    _p = len(cwow.a) / (len(cwow.a) + len(cwow.b))
    stats.cwow_final_percent = _p
    stats.cw_final_train = len(cw.a)
    stats.cw_final_valid = len(cw.b)
    stats.ow_final_train = len(ow.a)
    stats.ow_final_test = len(ow.b)
    log.info(f'final distribution: {len(cwow.a)} cw and '
             f'{len(cwow.b)} ow triples: {int(_p) * 100}% cw')

    path = ryn.ENV.SPLIT_DIR / name
    path.mkdir(exist_ok=True, parents=True)

    total = set.union(*[rel.triples for rel in rels])
    forgotten = total - cwow.unionized

    stats.forgotten = len(forgotten)

    check(rels, cw, ow, cwow, forgotten)
    write(path, g, cw, ow, forgotten)
    stats_entity_intersections(path, cw, ow)

    print('done')
    return stats


def create(
        g: graph.Graph,
        cfg: Config,
        rels: List[Relation],
        seeds: List[int]):

    for seed in seeds:
        name = f'{g.name.split("-")[0]}'
        name += f'_{cfg.owcw_split:.2f}-{cfg.trainvalid_split:.2f}'
        name += f'_{cfg.threshold}_{seed}'

        log.info(f'! creating dataset {name=}; set seed to {seed}')

        random.seed(seed)
        stats = _create(g, cfg, rels, name)
        stats.seed = seed

        yield stats
