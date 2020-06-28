# -*- coding: utf-8 -*-

"""

Create graph splits exposing tbox proxies.

"""

import ryn
from ryn.graphs import graph
from ryn.common import logging

import random
import pathlib
import textwrap
import operator
import functools

from dataclasses import dataclass

from typing import Any
from typing import Set
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union


log = logging.get('graph.split')


def _ents_from_triples(triples):
    hs, ts, _ = zip(*triples)
    return set(hs) | set(ts)


# ---


@dataclass
class Config:

    # split ratio (for example: retaining 70% of all
    # samples for training requires a value of .7)
    ow_split: float
    train_split: float

    # no of relation (sorted by ratio)
    threshold: int


@dataclass(eq=False)  # id based hashing
class Part:

    owe: Set[int]  # open world entities
    triples: Set[Tuple[int]]

    @property
    @functools.lru_cache
    def entities(self):
        return _ents_from_triples(self.triples)

    @property
    @functools.lru_cache
    def heads(self) -> Set[int]:
        return set(tuple(zip(*self.triples))[0])

    @property
    @functools.lru_cache
    def tails(self) -> Set[int]:
        return set(tuple(zip(*self.triples))[1])

    # ---

    def __str__(self) -> str:
        return (
            f'owe: {len(self.owe)}\n'
            f'triples: {len(self.triples)}\n'
            f'entities: {len(self.entities)}\n'
            f'heads: {len(self.heads)}\n'
            f'tails: {len(self.tails)}'
        )


@dataclass
class Dataset:
    """
    Container class for a split dataset

    """

    ent2id: Dict[int, str]
    rel2id: Dict[int, str]

    concepts: Set[int]

    cw_train: Part
    cw_valid: Part

    ow_train: Part
    ow_test: Part

    # ---

    def __str__(self) -> str:
        s = (
            'RYN SPLIT DATASET\n'
            f'-----------------\n'
            f'\n{len(self.concepts)} retained concepts\n\n'
        )

        # functools.partial not applicable :(
        def _indent(s):
            return textwrap.indent(s, '  ')

        s += f'\nClosed World - TRAIN:\n{_indent(str(self.cw_train))}'
        s += f'\nClosed World - VALID:\n{_indent(str(self.cw_valid))}'
        s += f'\nOpen World - TRAIN:\n{_indent(str(self.ow_train))}'
        s += f'\nOpen World - TEST:\n{_indent(str(self.ow_test))}'

        return s

    # ---

    def save(self, path: Union[str, pathlib.Path]):
        pass

    @classmethod
    def load(K, path: Union[str, pathlib.Path]) -> 'Dataset':
        """

        Load a dataset from disk

        Parameters
        ----------

        path : Union[str, pathlib.Path]
          Folder containing all necessary files

        """
        path = pathlib.Path(path)

        with (path / 'concepts.txt').open(mode='r') as fd:
            num = int(fd.readline())
            concepts = set(map(int, fd.readlines()))
            assert num == len(concepts)

        def _load_dict(filep):
            with filep.open(mode='r') as fd:
                num = int(fd.readline())
                gen = map(lambda l: l.rsplit(' ', maxsplit=1), fd.readlines())
                d = dict((int(val), key.strip()) for key, val in gen)

                assert num == len(d)
                return d

        ent2id = _load_dict(path / 'entity2id.txt')
        rel2id = _load_dict(path / 'relation2id.txt')

        owe = set()

        def _load_triples(fp) -> Part:
            nonlocal owe

            with fp.open(mode='r') as fd:
                num = int(fd.readline())
                triples = set(
                    tuple(map(int, line.split(' ')))
                    for line in fd.readlines()
                )

            assert num == len(triples)

            ents = _ents_from_triples(triples)
            part = Part(triples=triples, owe=ents - owe)
            owe |= ents

            return part

        dataset = K(
            concepts=concepts,
            ent2id=ent2id,
            rel2id=rel2id,
            cw_train=_load_triples(path / 'cw.train2id.txt'),
            cw_valid=_load_triples(path / 'cw.valid2id.txt'),
            ow_train=_load_triples(path / 'ow.train2id.txt'),
            ow_test=_load_triples(path / 'ow.test2id.txt'),
        )

        return dataset


# ---


@dataclass
class Relation:

    r: int
    name: str
    triples: Set[Tuple[int]]

    hs: Set[int]
    ts: Set[int]

    ratio: float

    @property
    def concepts(self) -> Set[int]:
        # either head or tail sets (whichever is smaller)
        reverse = len(self.hs) <= len(self.ts)
        return self.hs if reverse else self.ts

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


class Split:
    """
    Abstraction for disjunct sets of things.
    Used for both entity and triple sets.

    """

    def __getattr__(self, *args, **kwargs):
        (name, ) = args
        if name not in self._sets:
            raise AttributeError(f'"{name}" not found in Split')

        return self._sets[name]

    def __init__(self, **kwargs):
        assert all(type(v) is set for v in kwargs.values())
        self._sets = kwargs.copy()

    def _apply(self, op, other):
        try:
            assert self._sets.keys() == other._sets.keys()
            return Split(**{
                k: op(self._sets[k], other._sets[k])
                for k in self._sets})

        except AttributeError:
            return Split(**{k: op(self._sets[k], other) for k in self._sets})

    @property
    def lens(self):
        # in order of insertion
        return tuple(len(v) for v in self._sets.values())

    @property
    def unionized(self):
        return set.union(*self._sets.values())

    def check(self):
        union = self.unionized
        assert len(union) == 0, str(union)

    def intersection(self, other):
        return self._apply(operator.and_, other)

    def union(self, other):
        return self._apply(operator.or_, other)

    def without(self, other):
        return self._apply(operator.sub, other)

    def clone(self):
        return self._apply(operator.and_, self)


# ------------------------------------------------------------


@dataclass
class Splitter:

    name: str
    cfg: Config
    g: graph.Graph

    @property
    def rels(self) -> List[Relation]:
        return self._rels

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def __post_init__(self):
        self._rels = Relation.from_graph(self.g)
        self._rels.sort(key=lambda rel: rel.ratio)

        self._path = ryn.ENV.SPLIT_DIR / self.name
        self.path.mkdir(exist_ok=True, parents=True)

    def _partition(
            self,
            data: Set[Any],
            ratio: float) -> Tuple[Set[Any], Set[Any]]:

        lis = list(data)
        random.shuffle(lis)
        line = int(len(lis) * ratio)
        return set(lis[:line]), set(lis[line:])

    def create(self):
        print(f'create {self.name=}')

        concepts = self.rels[:self.cfg.threshold]
        concepts = set.union(*(rel.concepts for rel in concepts))
        print(f'{len(concepts)=}')

        candidates = list(set(self.g.source.ents) - concepts)
        random.shuffle(candidates)

        t = int(len(self.g.source.triples) * self.cfg.ow_split)
        _p = int(self.cfg.ow_split * 100)
        print(f'targeting {t=} triples for ow ({_p}%)')

        triples = Split(ow=set(), cw=set())
        while candidates and len(triples.ow) < t:
            e = candidates.pop()
            triples.ow |= self.g.find(heads={e}, tails={e})

        triples.cw = set(self.g.source.triples) - triples.ow
        print(f'retrieved {len(triples.ow)=} and {len(triples.cw)=} triples')

        # TODO sanity checks

        cw = Split(**dict(zip(
            ['train', 'valid'],
            self._partition(triples.cw, self.cfg.train_split)
        )))

        ow = Split(**dict(zip(
            ['train', 'test'],
            self._partition(triples.ow, self.cfg.train_split)
        )))

        print('writing')
        self.write(concepts, ow, cw)

    def write(self, concepts: Set[int], ow: Split, cw: Split):
        def _write(name, tups):
            with (self.path / name).open(mode='w') as fd:
                fd.write(f'{len(tups)}\n')
                lines = (' '.join(map(str, t)) for t in tups)
                fd.write('\n'.join(lines))
                log.info(f'wrote {fd.name}')

        _write('cw.train2id.txt', cw.train)
        _write('cw.valid2id.txt', cw.valid)
        _write('ow.train2id.txt', ow.train)
        _write('ow.test2id.txt', ow.test)

        _write(
            'relation2id.txt',
            [(v, k) for k, v in self.g.source.rels.items()])

        _write(
            'entity2id.txt',
            [(v, k) for k, v in self.g.source.ents.items()])

        _write('concepts.txt', [(e, ) for e in concepts])


def create(g: graph.Graph, cfg: Config, seeds: List[int]):
    for seed in seeds:
        name = f'{g.name.split("-")[0]}'
        name += f'_{cfg.ow_split:.2f}-{cfg.train_split:.2f}'
        name += f'_{cfg.threshold}_{seed}'

        log.info(f'! creating dataset {name=}; set seed to {seed}')

        random.seed(seed)

        gen = Splitter(g=g, cfg=cfg, name=name)
        gen.create()

        yield


def analyse(path: pathlib.Path):
    def _load_triples(fp) -> Set[Tuple[int]]:
        with fp.open(mode='r') as fd:
            fd.readline()
            return set(
                tuple(map(int, line.split(' ')))
                for line in fd.readlines()
            )

    with (path / 'concepts.txt').open(mode='r') as fd:
        fd.readline()
        concepts = set(int(e) for e in fd.readlines())

    triples = Split(
        cw=(
            _load_triples(path / 'cw.train2id.txt') |
            _load_triples(path / 'cw.valid2id.txt')),
        ow=(
            _load_triples(path / 'ow.train2id.txt') |
            _load_triples(path / 'ow.test2id.txt')), )

    entities = Split(
        ow=_ents_from_triples(triples.ow),
        cw=_ents_from_triples(triples.cw), )

    print()
    print(f'concepts: {len(concepts)}')
    print()

    print(f'cw triples:  {len(triples.cw):>7d}')
    print(f'ow triples:  {len(triples.ow):>7d}')
    print()

    print(f'ow entities: {len(entities.ow):>7d}')
    print(f'cw entities: {len(entities.cw):>7d}')
    print()

    print(f'entities cw - ow: {len(entities.cw - entities.ow)}')
    print(f'entities ow - cw: {len(entities.ow - entities.cw)}')
    print()

    _cw_linked = set(
        (h, t, r) for h, t, r in triples.cw
        if h in concepts or t in concepts)

    _ow_linked = set(
        (h, t, r) for h, t, r in triples.ow
        if h in concepts or t in concepts)

    print(f'cw triples linked to concepts: {len(_cw_linked)}')
    print(f'ow triples linked to concepts: {len(_ow_linked)}')
    print()
