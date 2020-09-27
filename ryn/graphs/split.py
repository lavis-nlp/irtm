# -*- coding: utf-8 -*-

"""

Create graph splits exposing tbox proxies.

"""

import ryn
from ryn.graphs import graph
from ryn.graphs import loader
from ryn.common import helper
from ryn.common import logging

import pickle
import random
import pathlib
import textwrap
import operator
import argparse

from datetime import datetime
from functools import partial
from functools import lru_cache
from dataclasses import dataclass
from itertools import combinations

from tqdm import tqdm as _tqdm

from typing import Set
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union


log = logging.get('graph.split')
tqdm = partial(_tqdm, ncols=80)


def _ents_from_triples(triples):
    hs, ts, _ = zip(*triples)
    return set(hs) | set(ts)


# ---


@dataclass
class Config:

    seed: int

    # split ratio (for example: retaining 70% of all
    # samples for training requires a value of .7)
    ow_split: float
    train_split: float

    # no of relation (sorted by ratio)
    threshold: int

    # post-init

    git: str = None  # revision hash
    date: datetime = None

    def __post_init__(self):
        self.git = helper.git_hash()
        self.date = datetime.now()

    def __str__(self) -> str:
        return 'Config:\n' + textwrap.indent((
            f'seed: {self.seed}\n'
            f'ow split: {self.ow_split}\n'
            f'train split: {self.train_split}\n'
            f'relation threshold: {self.threshold}\n'
            f'git: {self.git}\n'
            f'date: {self.date}\n'
        ), '  ')

    # ---

    def save(self, f_name: Union[str, pathlib.Path]):
        path = pathlib.Path(f_name)
        # _relative = path.relative_to(ryn.ENV.ROOT_DIR)
        # log.info(f'saving config to {_relative}')
        log.info(f'saving config to {path}')

        with path.open(mode='wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(f_name: Union[str, pathlib.Path]) -> 'Config':
        path = pathlib.Path(f_name)
        log.info(f'loading config from {path}')

        with path.open(mode='rb') as fd:
            return pickle.load(fd)


@dataclass(eq=False)  # id based hashing
class Part:

    name: str
    owe: Set[int]  # open world entities
    triples: Set[Tuple[int]]
    concepts: Set[int]

    @property
    @lru_cache
    def g(self) -> graph.Graph:
        return graph.Graph(source=graph.GraphImport(triples=self.triples))

    @property
    @lru_cache
    def entities(self):
        return _ents_from_triples(self.triples)

    @property
    @lru_cache
    def heads(self) -> Set[int]:
        return set(tuple(zip(*self.triples))[0])

    @property
    @lru_cache
    def tails(self) -> Set[int]:
        return set(tuple(zip(*self.triples))[1])

    @property
    @lru_cache
    def linked_concepts(self) -> Set[int]:
        return self.entities & self.concepts

    @property
    @lru_cache
    def concept_triples(self) -> Set[Tuple[int]]:
        g = graph.Graph(source=graph.GraphImport(triples=self.triples))
        return g.find(heads=self.concepts, tails=self.concepts)

    # ---

    def __str__(self) -> str:
        return (
            f'owe: {len(self.owe)}\n'
            f'triples: {len(self.triples)}\n'
            f'entities: {len(self.entities)}\n'
            f'heads: {len(self.heads)}\n'
            f'tails: {len(self.tails)}'
        )

    def __or__(self, other: 'Part') -> 'Part':
        return Part(
            name=f'{self.name}|{other.name}',
            owe=self.owe | other.owe,
            triples=self.triples | other.triples,
            concepts=self.concepts | other.concepts,
        )


@dataclass
class Dataset:
    """
    Container class for a split dataset

    """

    path: pathlib.Path

    cfg: Config
    g: graph.Graph

    id2ent: Dict[int, str]
    id2rel: Dict[int, str]

    concepts: Set[int]

    cw_train: Part
    cw_valid: Part

    ow_valid: Part
    ow_test: Part

    # ---

    @property
    def name(self):
        return self.path.name

    # ---

    def __str__(self) -> str:
        s = (
            'RYN.SPLIT DATASET\n'
            f'-----------------\n'
            f'\n{len(self.concepts)} retained concepts\n\n'
            f'{self.cfg}\n'
            f'{self.g.str_stats}\n'
            f'{self.path}\n'
        )

        # functools.partial not applicable :(
        def _indent(s):
            return textwrap.indent(s, '  ')

        s += f'\nClosed World - TRAIN:\n{_indent(str(self.cw_train))}'
        s += f'\nClosed World - VALID:\n{_indent(str(self.cw_valid))}'
        s += f'\nOpen World - VALID:\n{_indent(str(self.ow_valid))}'
        s += f'\nOpen World - TEST:\n{_indent(str(self.ow_test))}'

        return s

    def __getitem__(self, key: str):
        return {
            'cw.train': self.cw_train,
            'cw.valid': self.cw_valid,
            'ow.valid': self.ow_valid,
            'ow.test': self.ow_test,
        }[key]

    # ---

    def check(self):
        """

        Run some self diagnosis

        """
        log.info(f'! running self-check for {self.path.name} Dataset')

        # no triples must be shared between splits

        triplesets = (
            ('cw.train', self.cw_train.triples, ),
            ('cw.valid', self.cw_valid.triples, ),
            ('ow.valid', self.ow_valid.triples, ),
            ('ow.test', self.ow_test.triples, ),
        )

        for (n1, s1), (n2, s2) in combinations(triplesets, 2):
            assert s1.isdisjoint(s2), f'{n1} and {n2} share triples'

        # no ow entities must be shared between splits

        owesets = (
            ('cw.train', self.cw_train.owe, ),
            ('cw.valid', self.cw_valid.owe, ),
            ('ow.valid', self.ow_valid.owe, ),
            ('ow.test', self.ow_test.owe, ),
        )

        for (n1, s1), (n2, s2) in combinations(owesets, 2):
            assert s1.isdisjoint(s2), f'{n1} and {n2} share owe entities'

        # ow entities must not be seen in earlier splits
        # and no ow entities must occur in cw.valid
        # (use .entities property which gets this information directly
        # directly from the triple sets)

        assert self.cw_train.owe == self.cw_train.entities, (
            'cw.train owe != cw.train entities')
        assert not len(self.cw_valid.owe), (
            'cw.valid contains owe entities')

        seen = self.cw_train.entities | self.cw_valid.entities
        assert self.ow_valid.owe.isdisjoint(seen), (
            'entities in ow valid leaked')

        seen |= self.ow_valid.entities
        assert self.ow_test.owe.isdisjoint(seen), (
            'entities in ow test leaked)')

    @classmethod
    @helper.cached('.cached.dataset.pkl')
    def load(K, path: Union[str, pathlib.Path]) -> 'Dataset':
        """

        Load a dataset from disk

        Parameters
        ----------

        path : Union[str, pathlib.Path]
          Folder containing all necessary files

        """
        path = pathlib.Path(path)

        cfg = Config.load(path / 'cfg.pkl')
        g = graph.Graph.load(path / 'graph.pkl')

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

        id2ent = _load_dict(path / 'entity2id.txt')
        id2rel = _load_dict(path / 'relation2id.txt')

        def _load_part(fp, seen: Set[int], name: str) -> Part:
            nonlocal concepts

            with fp.open(mode='r') as fd:
                num = int(fd.readline())
                triples = set(
                    tuple(map(int, line.split(' ')))
                    for line in fd.readlines()
                )

            assert num == len(triples)

            ents = _ents_from_triples(triples)
            part = Part(
                name=name.replace('_', '.'),
                triples=triples,
                owe=ents - seen,
                concepts=concepts)

            return part

        parts = {}
        seen = set()

        for name, fp in (
                ('cw_train', path / 'cw.train2id.txt'),
                ('cw_valid', path / 'cw.valid2id.txt'),
                ('ow_valid', path / 'ow.valid2id.txt'),
                ('ow_test', path / 'ow.test2id.txt'), ):

            part = _load_part(fp, seen, name)
            seen |= part.entities
            parts[name] = part

        self = K(
            path=path,
            g=g, cfg=cfg,
            concepts=concepts,
            id2ent=id2ent, id2rel=id2rel,
            **parts)

        self.check()
        return self


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

    def create(self):
        """Create a new split dataset

        Three constraints apply:

        1. All concept entities must appear at least once in
           cw.train. Triples containing concept entities are
           distributed over all splits.

        2. The number of zero-shot entities needs to be maximised in
           ow: These entities are first encountered in their
           respective split; e.g. a zero-shot entity of ow.valid must
           not be present in any triple of cw.train oder cw.valid but
           may be part of an ow.test triples.

        3. The amount of triples should be balanced by the provided
           configuration (cfg.ow_split, cfg.train_split).

        """
        log.info(f'create {self.name=}')

        concepts = self.rels[:self.cfg.threshold]
        concepts = set.union(*(rel.concepts for rel in concepts))
        log.info(f'{len(concepts)=}')

        candidates = list(set(self.g.source.ents) - concepts)
        random.shuffle(candidates)

        _p = int(self.cfg.ow_split * 100)
        log.info(f'targeting {_p}% of all triples for ow')

        # there are three thresholds:
        # 0 < t1 < t2 < t3 < len(candidates) = n
        # where:
        #  0-t1:   ow valid
        #  t1-t2:  ow test
        #  t2-t3:  cw train
        #  t3-n:   cw valid

        t1 = self.cfg.ow_split * self.cfg.train_split
        t2 = self.cfg.ow_split
        t3 = t2 + (1 - t2) * self.cfg.train_split

        n = len(self.g.source.triples)
        t1, t2, t3 = (int(n * x) for x in (t1, t2, t3))
        log.info(f'target splits: 0 {t1=} {t2=} {t3=} {n=}')

        # retain all triples where both head and tail
        # are concept entities for cw train
        retained = self.g.select(heads=concepts, tails=concepts)
        agg = retained.copy()

        cw = Split(train=retained, valid=set())
        ow = Split(valid=set(), test=set())

        while candidates:
            e = candidates.pop()
            found = self.g.find(heads={e}, tails={e}) - agg

            agg |= found
            curr = len(agg)

            # open
            if curr < t1:
                ow.valid |= found
            elif curr < t2:
                ow.test |= found

            # closed
            elif curr < t3:
                cw.train |= found
            else:
                cw.valid |= found

        # ---

        assert len(agg) == len(self.g.source.triples)
        log.info(
            f'split {len(ow.unionized)=} and '
            f'{len(cw.unionized)=} triples')

        log.info('reordering triples')
        log.info(f'{len(cw.train)=} {len(cw.valid)=}')

        # ---

        # move all cw.valid triples containing
        # entities unseen in cw.train
        _n_train = len(cw.train)
        known = _ents_from_triples(cw.train)
        misplaced = set(filter(
            lambda trip: not all((trip[0] in known, trip[1] in known)),
            cw.valid))

        cw.train |= misplaced
        cw.valid -= misplaced

        log.info(f'! moved {len(cw.train) - _n_train} triples to cw train')
        log.info(f'{len(cw.train)=} {len(cw.valid)=}')

        log.info('writing')
        self.write(concepts, ow, cw)

    def write(self, concepts: Set[int], ow: Split, cw: Split):
        def _write(name, tups):
            with (self.path / name).open(mode='w') as fd:
                fd.write(f'{len(tups)}\n')
                lines = (' '.join(map(str, t)) for t in tups)
                fd.write('\n'.join(lines))

                _path = pathlib.Path(fd.name)
                _relative = _path.relative_to(ryn.ENV.ROOT_DIR)
                log.info(f'wrote {_relative}')

        _write('cw.train2id.txt', cw.train)
        _write('cw.valid2id.txt', cw.valid)
        _write('ow.valid2id.txt', ow.valid)
        _write('ow.test2id.txt', ow.test)

        _write(
            'relation2id.txt',
            [(v, k) for k, v in self.g.source.rels.items()])

        _write(
            'entity2id.txt',
            [(v, k) for k, v in self.g.source.ents.items()])

        _write('concepts.txt', [(e, ) for e in concepts])

        self.g.save(self.path / 'graph.pkl')
        self.cfg.save(self.path / 'cfg.pkl')


def create(g: graph.Graph, cfg: Config):
    name = f'{g.name.split("-")[0]}'
    name += f'_{cfg.seed}_{cfg.threshold}'

    log.info(f'! creating dataset {name=}')

    helper.seed(cfg.seed)
    Splitter(g=g, cfg=cfg, name=name).create()


def create_from_args(args: argparse.Namespace):
    assert args.uri, 'provide a graph uri'
    assert len(args.uri) == 1, 'provide a single graph uri'
    assert args.seeds, 'provide seeds'
    assert args.ratios, 'provide ratio thresholds'

    g = loader.load_graphs_from_uri(args.uri[0])[0]
    log.info(f'loaded {g.name}, analysing relations')

    rels = Relation.from_graph(g)
    rels.sort(key=lambda rel: rel.ratio)
    log.info(f'retrieved {len(rels)} relations')

    # if ow_split and train_split also become
    # parameters use itertools permutations

    print('')
    bar = tqdm(total=len(args.ratios) * len(args.seeds))

    K = partial(Config, ow_split=0.5, train_split=0.7)
    for threshold in args.ratios:
        log.info(f'! {threshold=}')

        for seed in args.seeds:
            log.info(f'! {seed=}')

            cfg = K(seed=seed, threshold=threshold)
            create(g, cfg)
            bar.update(1)


# ---


def _cli(args):
    import IPython

    print()
    if not args.path:
        raise ryn.RynError('please provide a --path')

    ds = Dataset.load(path=args.path)
    print(f'{ds}')

    banner = '\n'.join((
        '',
        '-' * 20,
        ' RYN DATASET CLIENT',
        '-' * 20,
        '',
        'variables in scope:',
        '    ds: Dataset',
        '',
    ))

    IPython.embed(banner1=banner)
