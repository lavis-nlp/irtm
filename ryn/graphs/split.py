# -*- coding: utf-8 -*-

"""

Create graph splits exposing tbox proxies.

"""

import ryn
from ryn.graphs import graph
from ryn.common import logging

import math
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


def _prob(x, a: float = 1, o: float = 1, s: float = 1):
    """

    Generalised logistic function

    Models the ratio distribution and allows for a better control
    of the "concept" retention policy while sampling.

    a controls the asymmetry
    o controls the offset
    s controls the stretch

    """

    # generalised logistic function
    return 1 / (1 + math.exp(s * (-x + o))) ** a


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

    pathname: str


@dataclass
class SoftConfig(Config):

    # see split.prob
    prob_a: float
    prob_o: float
    prob_s: float


@dataclass
class HardConfig(Config):

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

    train: Set[Any] = field(default_factory=set)
    valid: Set[Any] = field(default_factory=set)

    def _apply(self, op, other):
        try:
            return Split(
                train=op(self.train, other.train),
                valid=op(self.valid, other.valid), )
        except AttributeError:
            return Split(
                train=op(self.train, other),
                valid=op(self.valid, other), )

    @property
    def lens(self):
        return len(self.train), len(self.valid)

    @property
    def unionized(self):
        return self.train | self.valid

    def check(self):
        union = self.train & self.valid
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
        train = self.train
        valid = self.valid

        trainvalid = self.train & other.valid
        train -= trainvalid
        valid |= trainvalid

        validtrain = self.valid & other.train
        valid -= validtrain
        train |= validtrain

        split = Split(train=train, valid=valid)
        split.check()

        return split, len(trainvalid | validtrain)

    def partition(self, ratio: float) -> Tuple['Split', 'Split']:
        t1, t2 = _partition(self.train, int(len(self.train) * ratio))
        v1, v2 = _partition(self.valid, int(len(self.valid) * ratio))

        return Split(train=t1, valid=t2), Split(train=v1, valid=v2)


@dataclass
class Row:
    """
    Represents a row of the statistics gathered per relation type
    """

    rid: int
    ratio: int
    name: str

    p: int = -1

    concepts: int = -1
    retained_concepts: int = -1
    reordered_concepts: int = -1

    objects: int = -1
    retained_objects: int = -1
    reordered_objects: int = -1

    ow_triples: int = -1
    cw_triples: int = -1

    triples: int = -1
    retained_triples: int = -1

    HEADERS = (
        'id', 'ratio', 'p',
        'concepts', 'retained', 'reordered',
        'objects', 'retained', 'reordered',
        'ow triples', 'cw triples',
        'triples', 'retained',
        'name'
    )

    @property
    def tup(self):
        return (
            self.rid, self.ratio, self.p,
            self.concepts, self.retained_concepts, self.reordered_concepts,
            self.objects, self.retained_objects, self.reordered_objects,
            self.ow_triples, self.cw_triples,
            self.triples, self.retained_triples,
            self.name, )


def _ents_from_triples(ents):
    hs, ts, _ = zip(*list(ents))
    return set(hs) | set(ts)


def _track(s, x, name=''):
    if x in s.train:
        print(f'{x} found in {name} train')

    elif x in s.valid:
        print(f'{x} found in {name} valid')

    else:
        print(f'{x} found in neither train or valid')


# ---


def _split_concepts(
        i: int, cfg: Config, row: Row,
        concepts: Set[int], entities: Split):

    # using the probability function defined beforehand
    # to decide how many of the concept candidates remain
    # in the test set (50% to 100%)
    p = 1 - _prob(i, a=cfg.prob_a, o=cfg.prob_o, s=cfg.prob_s) * 0.5
    row.p = p

    # set threshold based on ratio
    line = int(p * len(concepts)) + 1
    train, valid = _partition(concepts, line)
    split = Split(train=train, valid=valid)

    # track(c_split, _N, 'c_split (pre-reorder)')
    split, reordered = split.reorder(entities)

    row.reordered_concepts = reordered / row.concepts
    row.retained_concepts = len(split.train) / row.concepts

    return split


def _split_objects(
        cfg: Config, row: Row,
        c_split: Split, objects: Set[int], entities: Split):

    # if x is in both concepts and objects
    # and ends up in either c_split.train or c_split
    # then it has to be put in the respective
    # o_split partition

    intersection = Split(
        train=objects & c_split.train,
        valid=objects & c_split.valid, )

    remains = objects - intersection.unionized
    N = int(cfg.trainvalid_split * len(remains))
    line = N + 1 - len(intersection.train)
    train, valid = _partition(remains, line)

    split = Split(train=train, valid=valid).union(intersection)
    split, reordered = split.reorder(entities)

    row.reordered_objects = reordered / row.objects
    row.retained_objects = len(split.train) / row.objects

    return split


def _select_triples(
        g: graph.Graph, row: Row,
        heads: Set[int], tails: Set[int],
        rel: Relation):

    select = partial(g.select, edges={rel.r})

    triples_train = select(heads=heads.train, tails=tails.train)
    triples_valid = select(heads=heads.valid, tails=tails.valid)

    t_selection = Split(train=triples_train, valid=triples_valid)

    row.triples = len(g.find(edges={rel.r}))
    row.retained_triples = len(t_selection.unionized) / row.triples

    return t_selection


# ---


def check(rels, cw, ow, triples, removed_triples):
    # triples is the union of cw and ow

    for s1, s2 in combinations((cw.train, cw.valid, ow.train, ow.valid), 2):
        assert not s1 & s2, f'{len(s1)=} {len(s2)=} {len(s1 & s2)=}'

    assert not removed_triples & triples.unionized

    _a = len(removed_triples | triples.unionized)
    _b = sum(len(rel.triples) for rel in rels)
    assert _a == _b, (
        f'removed + unionized: {_a} '
        f'original: {_b}')


# datasets are saved in OpenKE format
def write(
        path: pathlib.Path,
        g: graph.Graph,
        cw: Split, ow: Split,
        removed_triples: Set[Tuple[int]]):

    def _write(name, tups):
        with (path / name).open(mode='w') as fd:
            fd.write(f'{len(tups)}\n')
            lines = (' '.join(map(str, t)) for t in tups)
            fd.write('\n'.join(lines))
            log.info(f'wrote {fd.name}')

    _write('cw.train2id.txt', cw.train)
    _write('cw.valid2id.txt', cw.valid)
    _write('ow.train2id.txt', ow.train)
    _write('ow.test2id.txt', ow.valid)
    _write('removed2id.txt', removed_triples)

    _write('relation2id.txt', [(v, k) for k, v in g.source.rels.items()])
    _write('entity2id.txt', [(v, k) for k, v in g.source.ents.items()])


# ---


def _write_tsv(table, path, name):
    with (path / name).open(mode='w') as fd:
        fd.write(table(tablefmt='tsv'))
        log.info(f'wrote {fd.name}')


def _stats_rows(path, rows):
    table = partial(tabulate, [r.tup for r in rows], headers=Row.HEADERS)
    _write_tsv(table, path, 'stats.rels.tsv')


def _stats_entity_intersections(path, cw, ow):
    ents = (
        _ents_from_triples(cw.train),
        _ents_from_triples(cw.valid),
        _ents_from_triples(ow.train),
        _ents_from_triples(ow.valid), )

    names = 'cw.train', 'cw.valid', 'ow.train', 'ow.valid'

    intersections = [len(a & b) for a, b in product(ents, ents)]
    intersections = np.array(intersections).reshape((4, 4))

    rows = [
        (names[i], ) + tuple(intersections[i])
        for i in range(len(intersections))]

    table = partial(tabulate, rows, headers=names)
    _write_tsv(table, path, 'stats.ents.tsv')


def stats(path: pathlib.Path, rows: List[Row], cw, ow):

    log.info(f'! closed world: {len(cw.train):10d} {len(cw.valid):10d} ')
    log.info(f'!   open world: {len(ow.train):10d} {len(ow.valid):10d}')

    _stats_rows(path, rows)
    _stats_entity_intersections(path, cw, ow)


# ------------------------------------------------------------


def _create_soft(
        g: graph.Graph,
        cfg: Config,
        rels: List[Relation],
        name: str):

    rows = []

    entities = Split()
    triples = Split()
    removed_triples = set()

    # work by greedily start with the most obvious
    # concept candidates (i.e. high in- or out-degree)
    for i, rel in enumerate(sorted(rels, key=lambda rel: rel.ratio)):
        row = Row(rid=rel.r, ratio=rel.ratio, name=rel.name)

        # determine whether the range or domain of
        # the relations are assumed to be concepts
        reverse = len(rel.hs) <= len(rel.ts)
        concepts, objects = (rel.hs, rel.ts) if reverse else (rel.ts, rel.hs)

        row.concepts = len(concepts)
        row.objects = len(objects)

        c_split = _split_concepts(i, cfg, row, concepts, entities)
        o_split = _split_objects(cfg, row, c_split, objects, entities)
        e_union = c_split.union(o_split)

        heads, tails = (c_split, o_split) if reverse else (o_split, c_split)
        t_triples = _select_triples(g, row, heads, tails, rel)

        row.ow_triples = len(t_triples.train) / len(rel.triples)
        row.cw_triples = len(t_triples.valid) / len(rel.triples)

        # --------------------

        entities = entities.union(e_union)
        triples = triples.union(t_triples)
        removed_triples |= rel.triples - t_triples.unionized

        # --------------------

        try:
            e_union.check()
            triples.check()
            entities.check()
        except AssertionError as exc:
            # print(rel.r)
            # print(f'{_N in concepts=}')
            # print(f'{_N in objects=}')
            # _track(c_split, _N, 'c_split')
            # _track(c_split, _N, 'c_reordered')
            # _track(o_split, _N, 'o_split')
            # _track(o_split, _N, 'o_reordered')
            # _track(e_union, _N, 'e_union')
            raise exc

        rows.append(row)

    log.info('checking constraints')
    triples.check()
    entities.check()

    # partition closed/open world triple sets
    # into train/valid (ow) and train/test (cw)
    cw, ow = triples.partition(cfg.owcw_split)

    path = ryn.ENV.SPLIT_DIR / cfg.pathname / name
    path.mkdir(exist_ok=True, parents=True)

    check(rels, cw, ow, triples, removed_triples)
    write(path, g, cw, ow, removed_triples)
    stats(path, rows, cw, ow)


def create_soft(
        g: graph.Graph,
        cfg: SoftConfig,
        rels: List[Relation],
        seeds: List[int]):

    for seed in seeds:
        name = f'{g.name.split("-")[0]}'
        name += f'_{cfg.owcw_split:.2f}-{cfg.trainvalid_split:.2f}'
        name += f'_{seed}'

        log.info(f'! creating dataset {name=}; set seed to {seed}')

        random.seed(seed)
        _create_soft(g, cfg, rels, name)


# ------------------------------------------------------------

# method is used if the triple has to be put in either cw or ow
# because the entity is encountered in one of the splits
def _constrained_add(cwow: Split, entities: Split, triple):
    h, t, r = triple

    if h in entities.train and t in entities.train:
        cwow.train.add(triple)
        return None

    # select unencountered entity
    x = h if t in entities.train else t

    # if already in valid, add triple to ow
    if x in entities.valid:
        cwow.valid.add(triple)
        return None

    return x


def _incremental_add(cfg: HardConfig, cwow: Split, entities: Split, lis):
    while len(lis):
        triple = lis.pop()
        x = _constrained_add(cwow, entities, triple)
        if not x:
            continue

        # track whether the balance of ow/cw fits the configuration
        ratio = len(cwow.train) / (len(cwow.train) + len(cwow.valid))
        if ratio < cfg.owcw_split:
            assert x not in entities.valid, x
            entities.train.add(x)
            cwow.train.add(triple)

        else:
            assert x not in entities.train, x
            entities.valid.add(x)
            cwow.valid.add(triple)


def _create_hard(
        g: graph.Graph,
        cfg: HardConfig,
        rels: List[Relation],
        name: str):

    selection = rels[:cfg.threshold]
    retained = set()  # of entities

    for rel in selection:
        # determine whether the range or domain of
        # the relations are assumed to be concepts
        reverse = len(rel.hs) <= len(rel.ts)
        concepts, objects = (rel.hs, rel.ts) if reverse else (rel.ts, rel.hs)

        retained |= concepts

    remaining = set(g.source.triples.copy())
    log.info(f'found {len(retained)} concepts ({len(remaining)} remain)')

    cwow = Split()

    # triples where both head and tail are
    # retained must stay in the cw set

    cwow.train = g.select(heads=retained, tails=retained)
    remaining -= cwow.train
    log.info(f'moving {len(cwow.train)} triples to cw'
             f' ({len(remaining)} remain)')

    # find triples that are linked to the concepts
    candidates = list(g.find(heads=retained, tails=retained) & remaining)
    log.info(f'found {len(candidates)} triples linking concepts')

    # keep track of the entities such that no entities ends up in both cw/ow
    entities = Split(train=retained)

    random.shuffle(candidates)
    _incremental_add(cfg, cwow, entities, candidates)

    assert not len(candidates)
    entities.check()

    remaining -= cwow.unionized
    _p = int(len(cwow.train) / (len(cwow.train) + len(cwow.valid)) * 100)
    log.info(f'gathered {len(cwow.train)} cw and '
             f'{len(cwow.valid)} ow triples: '
             f'{_p}% cw ({len(remaining)} remaining)')

    _incremental_add(cfg, cwow, entities, remaining)

    assert not len(remaining)
    entities.check()

    log.info(f'final distribution: {len(cwow.train)} cw and '
             f'{len(cwow.valid)} ow triples: '
             f'{_p}% cw ({len(remaining)} remaining)')

    print('done')


def create_hard(
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
        _create_hard(g, cfg, rels, name)
