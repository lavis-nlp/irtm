# -*- coding: utf-8 -*-

"""

Create graph splits exposing tbox proxies.

"""

import ryn
from ryn.graphs import graph
from ryn.common import logging

import math
import random
import operator

from functools import partial
from dataclasses import field
from dataclasses import dataclass

from tabulate import tabulate

from typing import Any
from typing import Set
from typing import List
from typing import Tuple


log = logging.get('graph.split')


def prob(x, a: float = 1, o: float = 1, s: float = 1):
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


@dataclass
class Config:

    # split ratio (for example: retaining 70% of all
    # samples for training requires a value of .7)
    split: float

    # see split.prob
    prob_a: float
    prob_o: float
    prob_s: float


@dataclass
class Relation:

    r: int
    name: str

    hs: Set[int]
    ts: Set[int]

    ratio: float

    @classmethod
    def from_graph(K, g: graph.Graph) -> List['Relation']:
        rels = []
        for r, relname in g.source.rels.items():
            hs, ts = map(set, zip(*((h, t) for h, t, _ in g.find(edges={r}))))
            lens = len(hs), len(ts)
            ratio = min(lens) / max(lens)

            rels.append(K(
                r=r, name=relname,
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

    def reorder(self, other):
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

    triples: int = -1
    retained_triples: int = -1

    HEADERS = (
        'id', 'ratio', 'p',
        'concepts', 'retained', 'reordered',
        'objects', 'retained', 'reordered',
        'triples', 'retained',
        'name'
    )

    @property
    def tup(self):
        return (
            self.rid, self.ratio, self.p,
            self.concepts, self.retained_concepts, self.reordered_concepts,
            self.objects, self.retained_objects, self.reordered_objects,
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


def _partition(data: Set[Any], line: int) -> Tuple[Set[Any], Set[Any]]:
    lis = list(data)
    random.shuffle(lis)
    return set(lis[:line]), set(lis[line:])


def _split_concepts(
        i: int, cfg: Config, row: Row,
        concepts: Set[int], entities: Split):

    # using the probability function defined beforehand
    # to decide how many of the concept candidates remain
    # in the test set (50% to 100%)
    p = 1 - prob(i) * 0.5
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
    line = int(cfg.split * len(remains)) + 1 - len(intersection.train)
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


def _create(g: graph.Graph, cfg: Config, rels: List[Relation], name: str):
    rows = []

    entities = Split()
    triples = Split()

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

        # --------------------

        triples = triples.union(t_triples)
        entities = entities.union(e_union)

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

    # --------------------

    table = partial(tabulate, [r.tup for r in rows], headers=Row.HEADERS)
    # print(table())

    path = ryn.ENV.CACHE_DIR / 'notes.graph.split' / name
    path.mkdir(exist_ok=True, parents=True)

    with (path / 'stats.tsv').open(mode='w') as fd:
        fd.write(table(tablefmt='tsv'))

    # log.info(f'created {len(train)=} {len(valid)=}')

    # save oke-like
    # with (path / 'train2id.txt').open(mode='w') as fd:
    #     fd.write(f'{len(train)}\n')
    #     fd.write('\n'.join(' '.join(map(str, triple)) for triple in train))
    #     print(f'wrote {fd.name}')

    # with (path / 'valid2id.txt').open(mode='w') as fd:
    #     fd.write(f'{len(valid)}\n')
    #     fd.write('\n'.join(' '.join(map(str, triple)) for triple in valid))
    #     print(f'wrote {fd.name}')

    # with (path / 'entity2id.txt').open(mode='w') as fd:
    #     fd.write(f'{len(g.source.ents)}\n')
    #     fd.write('\n'.join(
    #         f'{name} {eid}' for eid, name in g.source.ents.items()))
    #     print(f'wrote {fd.name}')

    # with (path / 'relation2id.txt').open(mode='w') as fd:
    #     fd.write(f'{len(g.source.rels)}\n')
    #     fd.write('\n'.join(
    #         f'{name} {rid}' for rid, name in g.source.rels.items()))
    #     print(f'wrote {fd.name}')


def create(
        g: graph.Graph,
        cfg: Config,
        rels: List[Relation],
        seeds: List[int]):

    for seed in seeds:
        name = f'{g.name.split("-")[0]}_{cfg.split:.2f}_{seed}'

        log.info(f'! creating dataset {name=}')
        random.seed(seed)
        _create(g, cfg, rels, name)
