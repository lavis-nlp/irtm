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
import contextlib

from datetime import datetime
from functools import partial
from functools import lru_cache
from dataclasses import dataclass
from collections import defaultdict

import transformers as tf
from nltk.tokenize import sent_tokenize as split_sentences

from typing import IO
from typing import List
from typing import Dict
from typing import Union


log = logging.get('text.encoder')


class Tokenizer:

    @property
    @lru_cache
    def vocab(self) -> Dict[int, str]:
        return {v: k for k, v in self._tok.vocab.items()}

    @helper.notnone
    def __init__(self, model: str = None):
        cache_dir = str(ryn.ENV.CACHE_DIR / 'lib.transformers')
        self._tok = tf.BertTokenizer.from_pretrained(
            model, cache_dir=cache_dir)

    @helper.notnone
    def __call__(self, sentences: List[str] = None, **kwargs):

        # sets 'input_ids', 'token_type_ids', 'attention_mask'
        res = self._tok(sentences, **kwargs)
        decoded = [self._tok.decode(ids) for ids in res['input_ids']]
        return {**res, **{'decoded': decoded}}


# ---


@dataclass
class TransformContext:

    select: loader.SQLite

    fd_indexes: IO[str]
    fd_tokenized: IO[str]
    fd_nocontext: IO[str]

    tokenizer: Tokenizer

    dataset: split.Dataset
    sentences: int
    tokens: int


def _transform_split(ctx: TransformContext, part: split.Part):
    log.info(f'transforming split {part.name}')

    # ---

    def _ints2str(lis: List[int]):
        return ' '.join(map(str, lis))

    # ---

    log.info(f'setting seed to {ctx.dataset.cfg.seed}')
    random.seed(ctx.dataset.cfg.seed)

    # init text files

    ctx.fd_indexes.write(
        '# Format: <ID>, <INDEX1> <INDEX2> ...\n'.encode())
    ctx.fd_tokenized.write(
        '# Format: <ID>, <NAME>, <TOKEN1> <TOKEN2> ...\n'.encode())
    ctx.fd_nocontext.write(
        '# Format: <ID>, <NAME>\n'.encode())

    ents = list(part.entities)
    shape = len(ents), ctx.sentences, 3, ctx.tokens

    # iterate entities

    gen = list((i, e, ctx.dataset.id2ent[e]) for i, e in enumerate(ents))
    assert (len(gen) == shape[0]) and (shape[0] == len(ents))

    for i, e, name in helper.tqdm(gen):
        result = ctx.select.by_entity(name)  # TODO replace with id selection
        if not result:
            log.info(f'! no contexts found for {e}: {name}')
            ctx.fd_nocontext.write(f'{e}, {name}\n'.encode())
            continue

        texts = list(zip(*result))[1]

        # process text
        # TODO remove; then do '\n'.join(...).split('\n')

        texts = map(split_sentences, texts)
        sentences = [t for sub in texts for t in sub]

        # select text

        random.shuffle(sentences)
        sentences = sentences[:ctx.sentences]

        # tokenize and map to vocabulary ids

        toks = ctx.tokenizer(
            sentences=sentences,
            padding=False,
            max_length=ctx.tokens,
            truncation=True)

        # write clear text

        # you cannot use 'decoded' for is_tokenized=True
        tokens = (
            ' '.join(ctx.tokenizer.vocab[idx] for idx in sentence)
            for sentence in toks['input_ids'])

        ctx.fd_tokenized.write(('\n'.join(
            f'{e}, {name}, {t}'
            for t in tokens) + '\n').encode())

        ctx.fd_indexes.write(('\n'.join(
            f'{e}, '
            f'{_ints2str(toks["input_ids"][i])}'
            for i in range(len(sentences))) + '\n').encode())


@helper.notnone
def transform(
        *,
        dataset: split.Dataset = None,
        database: str = None,
        sentences: int = None,
        tokens: int = None,
        model: str = None):
    """

    Tokenize and map the text for a dataset

    The produced files can be read by text.model.Data

    """

    name = f'{model}.{sentences}.{tokens}'
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

    # --

    def _get_ctx(stack, split: str):
        gopen = partial(gzip.open, mode='wb')

        ctx = TransformContext(

            fd_tokenized=stack.enter_context(
                gopen(str(p_out / f'{split}-tokenized.txt.gz'))),
            fd_indexes=stack.enter_context(
                gopen(str(p_out / f'{split}-indexes.txt.gz'))),
            fd_nocontext=stack.enter_context(
                gopen(str(p_out / f'{split}-nocontext.txt.gz'))),

            select=stack.enter_context(
                loader.SQLite(database=database)),

            tokenizer=Tokenizer(model=model),
            dataset=dataset,
            sentences=sentences,
            tokens=tokens,
        )

        return ctx

    with contextlib.ExitStack() as stack:

        print('\nclosed world - train:')
        log.info('! transforming: closed world - train')
        _transform_split(_get_ctx(stack, 'cw.train'), dataset.cw_train)

        print('\nclosed world - valid:')
        log.info('! transforming: closed world - valid')
        _transform_split(_get_ctx(stack, 'cw.valid'), dataset.cw_valid)

        print('\nopen world - valid:')
        log.info('! transforming: open world - valid')
        _transform_split(_get_ctx(stack, 'ow.valid'), dataset.ow_valid)

        print('\nopen world - test:')
        log.info('! transforming: open world - test')
        _transform_split(_get_ctx(stack, 'ow.test'), dataset.ow_test)


def _transform_from_args(args):
    assert args.dataset, 'provide --dataset'
    assert args.database, 'provide --database'
    assert args.sentences, 'provide --sentences'
    assert args.tokens, 'provide --tokens'
    assert args.model, 'provide --model'

    dataset = split.Dataset.load(args.dataset)
    transform(
        dataset=dataset,
        database=args.database,
        sentences=args.sentences,
        tokens=args.tokens,
        model=args.model, )


@dataclass
class Part:
    """
    Data for a specific split

    Initialized in model.Data

    """

    name: str  # e.g. cw.train (matches split.Part.name)
    no_context: Dict[int, str]

    id2toks: Dict[int, List[str]]
    id2ent: Dict[int, str]

    @classmethod
    @helper.notnone
    def load(K, *, name: str = None, path: pathlib.Path = None):
        log.info(f'! loading text dataset from {path}')

        # tokens
        tokens_path = path / f'{name}-tokenized.txt.gz'
        with gzip.open(str(tokens_path), mode='r') as fd:
            fd.readline()  # skip head comment
            id2ent, id2toks = {}, defaultdict(list)

            for line in map(bytes.decode, fd.readlines()):
                parts = line.split(',', maxsplit=2)
                e_str, e_name, tokens = map(str.strip, parts)
                e = int(e_str)

                id2ent[e] = e_name
                id2toks[e].append(tokens)

        log.info(f'loaded {len(id2toks)} entities and their sentences')

        # no contexts

        no_context_path = path / f'{name}-nocontext.txt.gz'
        with gzip.open(str(no_context_path), mode='r') as fd:
            fd.readline()  # skip head comment
            lines = fd.read().decode().strip().split('\n')
            items = (line.split(',', maxsplit=1) for line in lines)

            no_context = {int(k): v for k, v in items}

        log.info(
            f'loaded {len(no_context)} contextless entities '
            f'from {no_context_path}')

        return K(
            name=name,
            id2toks=id2toks, id2ent=id2ent,
            no_context=no_context)


@dataclass
class Dataset:
    """

    Tokenized text data ready to be used by a model

    This data is produced by ryn.text.encoder.transform.
    Files required for loading:

      - info.json
      - idxs.h5
      - <SPLIT>-nocontext.txt.gz

    """

    created: datetime
    git_hash: str

    model: str
    dataset: str
    database: str

    sentences: int
    tokens: int

    cw_train: Part
    cw_valid: Part
    ow_valid: Part
    ow_test: Part

    @property
    def name(self) -> str:
        return (
            f'{self.dataset}/{self.database}/'
            f'{self.model}.{self.sentences}.{self.tokens}')

    def close(self):
        log.info(f'closing model.Data for {self.name}')
        self.h5fd.close()

    @classmethod
    @helper.notnone
    def load(K, *, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)

        with (path / 'info.json').open(mode='r') as fd:
            info = json.load(fd)

        data = Dataset(
            created=datetime.fromisoformat(info['created']),
            git_hash=info['git_hash'],
            model=info['model'],
            dataset=info['dataset'],
            database=info['database'],
            sentences=info['sentences'],
            tokens=info['tokens'],

            cw_train=Part.load(name='cw.train', path=path),
            cw_valid=Part.load(name='cw.valid', path=path),
            ow_valid=Part.load(name='ow.valid', path=path),
            ow_test=Part.load(name='ow.test', path=path),
        )

        log.info(f'loaded {data.name}')

        return data
