# -*- coding: utf-8 -*-

import ryn

from ryn.text import loader
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import gzip
import json
import random
import pathlib
import textwrap
import contextlib
import multiprocessing as mp

from datetime import datetime
from functools import partial
from functools import lru_cache
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

import transformers as tf
from nltk.tokenize import sent_tokenize as split_sentences

from typing import IO
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple

log = logging.get('text.data')


SEP = ' • '


class Tokenizer:

    @property
    def base(self):
        return self._base

    @property
    @lru_cache
    def vocab(self) -> Dict[int, str]:
        return {v: k for k, v in self.base.vocab.items()}

    @helper.notnone
    def __init__(self, model: str = None):
        cache_dir = str(ryn.ENV.CACHE_DIR / 'lib.transformers')
        self._base = tf.BertTokenizer.from_pretrained(
            model, cache_dir=cache_dir)

# ---


@dataclass
class TransformContext:

    select: loader.SQLite

    fd_indexes: IO[str]
    fd_sentences: IO[str]
    fd_tokens: IO[str]
    fd_nocontext: IO[str]

    tokenizer: Tokenizer

    dataset: split.Dataset
    sentences: int
    tokens: int


def _split_sentences(blob: str, mention: str) -> Tuple[str]:
    blob = ' '.join(blob.split())
    split = split_sentences(blob)
    sel = tuple(
        s.replace('[MASK]', mention)
        for s in split if '[MASK]' in s)

    return sel


def _transform_result(result, *, e: int = None, amount: int = None):
    entities, mentions, blobs = zip(*result)
    assert all(e == db_entity for db_entity in entities)

    # TODO adjust after next db update
    # ----------
    mult = [
        _split_sentences(blob, mention)
        for blob, mention in zip(blobs, mentions)]

    sentences = [
        sentence
        for res in mult for sentence in res
        if sentence]

    if not sentences:
        return None

    # ----------

    # select text

    random.shuffle(sentences)
    sentences = tuple(
        # remove unnecessary whitespace
        ' '.join(sentence.split()) for sentence in
        sentences[:amount])

    return sentences


def _transform_split(wid: int, ctx: TransformContext, part: split.Part):
    log.info(f'transforming split {part.name}')

    # ---

    helper.seed(ctx.dataset.cfg.seed)

    # init text files

    ctx.fd_indexes.write(
        f'# Format: <ID>{SEP}<INDEX1> <INDEX2> ...\n'.encode())
    ctx.fd_sentences.write(
        f'# Format: <ID>{SEP}<NAME>{SEP}<SENTENCE>\n'.encode())
    ctx.fd_tokens.write(
        f'# Format: <ID>{SEP}<NAME>{SEP}<TOKEN1> <TOKEN2> ...\n'.encode())
    ctx.fd_nocontext.write(
        f'# Format: <ID>{SEP}<NAME>\n'.encode())

    ents = list(part.owe)
    shape = len(ents), ctx.sentences, 3, ctx.tokens

    # iterate entities

    def _ints2str(lis: List[int]):
        return ' '.join(map(str, lis))

    def _write(fd, s: str):
        assert '\n' not in s, s
        fd.write((s + '\n').encode())

    gen = list((i, e, ctx.dataset.id2ent[e]) for i, e in enumerate(ents))
    assert (len(gen) == shape[0]) and (shape[0] == len(ents))

    bar = partial(
        helper.tqdm,
        position=wid,
        desc=part.name,
        unit=' entities',)

    for i, e, name in bar(gen):
        result = ctx.select.by_entity(e)

        def _log(msg: str):
            log.info(f'! {msg} {e}: {name}')

        if not result:
            _log('no contexts found for')
            _write(ctx.fd_nocontext, f'{e}{SEP}{name}')
            continue

        sentences = _transform_result(result, e=e, amount=ctx.sentences)

        if not sentences:
            _log('no sentences with [MASK] found for')
            _write(ctx.fd_nocontext, f'{e}{SEP}{name}')
            continue

        # tokenize and map to vocabulary ids

        indexes = ctx.tokenizer.base(
            sentences,
            padding=False,
            max_length=ctx.tokens,
            truncation=True)['input_ids']

        # write clear text
        # you cannot use 'decoded' for is_tokenized=True

        assert len(indexes) == len(sentences), (
            f'{len(indexes)=} != {len(sentences)=}')

        for sentence, idx_list in zip(sentences, indexes):
            tokens = ' '.join(ctx.tokenizer.vocab[idx] for idx in idx_list)
            idxstr = ' '.join(map(str, idx_list))

            if not all((sentence, tokens, idxstr)):
                log.error(f'skipping empty sentence of {e} ({name})')
                continue

            _write(ctx.fd_sentences, f'{e}{SEP}{name}{SEP}{sentence}')
            _write(ctx.fd_tokens, f'{e}{SEP}{name}{SEP}{tokens}')
            _write(ctx.fd_indexes, f'{e}{SEP}{idxstr}')


@dataclass
class WorkerArgs:

    model: str
    tokens: int
    database: str
    sentences: int

    p_out: pathlib.Path
    dataset: split.Dataset


def _transform_get_ctx(stack, split: str, args: WorkerArgs):
    gopen = partial(gzip.open, mode='wb')

    ctx = TransformContext(

        fd_tokens=stack.enter_context(
            gopen(str(args.p_out / f'{split}-tokens.txt.gz'))),
        fd_sentences=stack.enter_context(
            gopen(str(args.p_out / f'{split}-sentences.txt.gz'))),
        fd_indexes=stack.enter_context(
            gopen(str(args.p_out / f'{split}-indexes.txt.gz'))),
        fd_nocontext=stack.enter_context(
            gopen(str(args.p_out / f'{split}-nocontext.txt.gz'))),

        select=stack.enter_context(
            loader.SQLite(database=args.database, to_memory=True)),

        tokenizer=Tokenizer(model=args.model),
        dataset=args.dataset,
        sentences=args.sentences,
        tokens=args.tokens,
    )

    return ctx


def _transform_worker(packed):
    wid, split, args = packed

    with contextlib.ExitStack() as stack:
        log.info(f'! dispatching worker #{wid} for {split}')

        part = args.dataset[split]
        ctx = _transform_get_ctx(stack, split, args)
        _transform_split(wid, ctx, part)


def transform(
        *,
        dataset: split.Dataset = None,
        database: str = None,
        sentences: int = None,
        tokens: int = None,
        model: str = None,
        suffix: str = None, ):
    """

    Tokenize and map the text for a dataset

    The produced files can be read by text.model.Data

    """

    # cannot save that to a csv
    _conflicts = set(name for name in dataset.id2ent.values() if SEP in name)
    if _conflicts:
        raise ryn.RynError(f'entities contain "{SEP}": {_conflicts}')

    # ---

    name = f'{model}.{sentences}.{tokens}'
    if suffix:
        name = f'{name}-{suffix}'

    p_out = ryn.ENV.TEXT_DIR / 'data' / dataset.name / name
    p_out.mkdir(exist_ok=True, parents=True)

    with (p_out / 'info.json').open(mode='w') as fd:

        info = dict(
            created=datetime.now().isoformat(),
            git_hash=helper.git_hash(),
            dataset=dataset.name,
            database=pathlib.Path(database).name,
            sentences=sentences,
            tokens=tokens,
            model=model,
        )

        json.dump(info, fd, indent=2)

    with mp.Pool(processes=4) as pool:
        args = WorkerArgs(
            p_out=p_out,
            dataset=dataset,
            database=database,
            sentences=sentences,
            tokens=tokens,
            model=model,
        )

        pool.map(_transform_worker, [
            (1, 'cw.train', args),
            (2, 'ow.valid', args),
            (3, 'ow.test', args),
        ])


def _transform_from_args(args):
    assert args.dataset, 'provide --dataset'
    assert args.database, 'provide --database'
    assert args.sentences, 'provide --sentences'
    assert args.tokens, 'provide --tokens'
    assert args.model, 'provide --model'

    dataset = split.Dataset.load(path=args.dataset)
    transform(
        dataset=dataset,
        database=args.database,
        sentences=args.sentences,
        tokens=args.tokens,
        model=args.model,
        suffix=args.suffix, )


@dataclass
class Part:
    """
    Data for a specific split

    """

    name: str  # e.g. cw.train-SUFFIX (matches split.Part.name)

    id2toks: Dict[int, List[str]] = field(default_factory=dict)
    id2idxs: Dict[int, List[int]] = field(default_factory=dict)
    id2sents: Dict[int, List[str]] = field(default_factory=dict)

    id2ent: Dict[int, str] = field(default_factory=dict)

    def __or__(self, other: 'Part') -> 'Part':
        return Part(
            name=f'{self.name}|{other.name}',
            id2sents={**self.id2sents, **other.id2sents},
            id2toks={**self.id2toks, **other.id2toks},
            id2idxs={**self.id2idxs, **other.id2idxs},
            id2ent={**self.id2ent, **other.id2ent},
        )

    def __str__(self) -> str:
        summed = sum(len(sents) for sents in self.id2sents.values())
        avg = (summed / len(self.id2sents)) if len(self.id2sents) else 0

        return '\n'.join((
            f'Part: {self.name}',
            f'  total entities: {len(self.id2ent)}',
            f'  average sentences per entity: {avg:2.2f}',
        ))

    def check(self):
        log.info(f'running self check for {self.name}')

        assert len(self.id2sents) == len(self.id2ent), (
            f'{len(self.id2sents)=} != {len(self.id2ent)=}')
        assert len(self.id2sents) == len(self.id2toks), (
            f'{len(self.id2sents)=} != {len(self.id2toks)=}')
        assert len(self.id2sents) == len(self.id2idxs), (
            f'{len(self.id2sents)=} != {len(self.id2idxs)=}')

        def _deep_check(d1: Dict[int, List], d2: Dict[int, List]):
            for e in d1:
                assert len(d1[e]) == len(d2[e]), (
                    f'{len(d1[e])=} != {len(d2[e])=}')

        _deep_check(self.id2sents, self.id2toks)
        _deep_check(self.id2sents, self.id2idxs)

    @helper.notnone
    def split_by_entity(self, *, ratio: float = None):
        n = int(len(self.id2idxs) * ratio)

        log.info(
            f'split {self.name} by entity at '
            f'{n}/{len(self.id2idxs)} ({ratio=})')

        def _split_dic(dic: Dict) -> Tuple[Dict]:
            lis = list(dic.items())
            return dict(lis[:n]), dict(lis[n:])

        kwargs_split = dict(
            id2toks=_split_dic(self.id2toks),
            id2idxs=_split_dic(self.id2idxs),
            id2sents=_split_dic(self.id2sents),
        )

        kwargs_a = {k: v[0] for k, v in kwargs_split.items()}
        kwargs_b = {k: v[1] for k, v in kwargs_split.items()}

        def _refine_kwargs(kwargs, suffix: str):
            return {
                **kwargs,
                **dict(
                    name=self.name + suffix,
                    id2ent={e: self.id2ent[e] for e in kwargs['id2idxs']}
                )
            }

        p1 = Part(**_refine_kwargs(kwargs_a, '-entity_split_a'))
        p2 = Part(**_refine_kwargs(kwargs_b, '-entity_split_b'))

        p1.check()
        p2.check()

        return p1, p2

    @helper.notnone
    def split_text_contexts(self, *, ratio: float = None):
        log.info(f'split {self.name} text contexts ({ratio=})')

        p1 = Part(name=self.name + '-context_split_a')
        p2 = Part(name=self.name + '-context_split_b')

        def _split(lis):
            n = int(len(lis) * ratio) + 1
            return lis[:n], lis[n:]

        for e in self.id2ent:
            id2toks = _split(self.id2toks[e])
            id2idxs = _split(self.id2idxs[e])
            id2sents = _split(self.id2sents[e])

            assert len(id2toks[0]), f'{e}: {self.id2toks[e]}'

            p1.id2toks[e] = id2toks[0]
            p1.id2idxs[e] = id2idxs[0]
            p1.id2sents[e] = id2sents[0]
            p1.id2ent[e] = self.id2ent[e]

            if len(id2toks[1]):
                p2.id2toks[e] = id2toks[1]
                p2.id2idxs[e] = id2idxs[1]
                p2.id2sents[e] = id2sents[1]
                p2.id2ent[e] = self.id2ent[e]

        p1.check()
        p2.check()

        _n = len(p1.id2ent) - len(p2.id2ent)
        log.info(f'finished split: {_n} have no validation contexts')
        return p1, p2

    # ---

    @staticmethod
    def _read(path, fname):
        with gzip.open(str(path / fname), mode='r') as fd:
            log.info(f'reading {path.name}/{fname}')
            fd.readline()  # consume head comment

            for line in map(bytes.decode, fd.readlines()):
                e_str, blob = line.split(SEP, maxsplit=1)
                yield int(e_str), blob

    @classmethod
    @helper.notnone
    def load(K, *, name: str = None, path: pathlib.Path = None):
        log.info(f'loading dataset from {path}')

        read = partial(Part._read, path)
        id2ent = {}

        # sentences
        id2sents = defaultdict(list)
        for e, blob in read(f'{name}-sentences.txt.gz'):
            e_name, sentence = blob.split(SEP, maxsplit=1)
            id2sents[e].append(sentence)
            id2ent[e] = e_name

        # tokens
        id2toks = defaultdict(list)
        for e, blob in read(f'{name}-tokens.txt.gz'):
            _, tokens = blob.split(SEP, maxsplit=1)
            id2toks[e].append(tuple(tokens.split()))

        # # indexes
        id2idxs = defaultdict(list)
        for e, blob in read(f'{name}-indexes.txt.gz'):
            id2idxs[e].append(tuple(map(int, blob.split())))

        log.info(f'loaded data for {len(id2ent)} distinct entities')

        part = K(
            name=name,
            id2ent=id2ent,
            id2toks=id2toks,
            id2idxs=id2idxs,
            id2sents=id2sents, )

        part.check()
        return part


@dataclass
class Dataset:
    """

    Tokenized text data ready to be used by a model

    This data is produced by ryn.text.data.transform and is
    reflecting the data as seen by ryn.graph.split.Dataset.

    Files required for loading:

      - info.json
      - <SPLIT>-indexes.txt.gz
      - <SPLIT>-sentences.txt.g
      - <SPLIT>-tokens.txt.gz

    """

    created: datetime
    git_hash: str

    model: str
    dataset: str
    database: str

    max_sentence_count: int
    max_token_count: int

    train: Part
    inductive: Part
    transductive: Part
    test: Part

    @property
    def name(self) -> str:
        return (
            f'{self.dataset}/{self.database}/{self.model}'
            f'.{self.max_sentence_count}.{self.max_token_count}')

    def __str__(self) -> str:
        buf = '\n'.join((
            'ryn.text.data.Dataset',
            f'{self.name}',
            f'created: {self.created}',
            f'git_hash: {self.git_hash}',
            '',
        ))

        for part in (
                self.cw_train,
                self.ow_valid,
                self.ow_test):

            buf += textwrap.indent(str(part), '  ') + '\n'

        return buf

    def check(self):
        log.info(f'running self check for {self.name}')

        def _disjoint(a, b):
            return not (len(set(a) & set(b)) > 0)

        assert _disjoint(self.train.id2ent, self.inductive.id2ent)
        assert _disjoint(self.train.id2toks, self.inductive.id2toks)
        assert _disjoint(self.train.id2idxs, self.inductive.id2idxs)
        assert _disjoint(self.train.id2sents, self.inductive.id2sents)

        def _sub(a, b):
            return set(a).issubset(set(b))

        assert _sub(self.transductive.id2ent, self.train.id2ent)
        assert _sub(self.transductive.id2toks, self.train.id2toks)
        assert _sub(self.transductive.id2idxs, self.train.id2idxs)
        assert _sub(self.transductive.id2sents, self.train.id2sents)

    @classmethod
    @helper.notnone
    @helper.cached('.cached.data.dataset.pkl')
    def load(K, path: Union[str, pathlib.Path], ratio: float = None):

        path = pathlib.Path(path)
        if not path.is_dir():
            raise ryn.RynError(f'Dataset cannot be found: {path}')

        with (path / 'info.json').open(mode='r') as fd:
            info = json.load(fd)

        # create splits
        train = Part.load(name='cw.train', path=path)
        train, transductive = train.split_text_contexts(ratio=ratio)
        inductive = Part.load(name='ow.valid', path=path)
        inductive, test = inductive.split_by_entity(ratio=ratio)

        self = Dataset(
            # update
            created=datetime.now().isoformat(),
            git_hash=helper.git_hash(),

            # copy
            model=info['model'],
            dataset=info['dataset'],
            database=info['database'],
            max_sentence_count=info['sentences'],
            max_token_count=info['tokens'],

            # create
            train=train,
            inductive=inductive,
            transductive=transductive,
            test=test,
        )

        self.check()

        log.info(f'loaded {self.name}')
        return self


def _cli(args):
    import IPython

    print()
    if not args.dataset:
        raise ryn.RynError('please provide a --dataset')

    ds = Dataset.load(path=args.dataset)
    print(f'{ds}')

    banner = '\n'.join((
        '',
        '-' * 20,
        ' RYN KEEN CLIENT',
        '-' * 20,
        '',
        'variables in scope:',
        '    ds: Dataset',
        '',
    ))

    IPython.embed(banner1=banner)
