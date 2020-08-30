# -*- coding: utf-8 -*-


import ryn

from ryn.text import loader
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import gzip
import random
import pathlib
import contextlib

from datetime import datetime
from dataclasses import dataclass

import h5py
import transformers as tf
from nltk.tokenize import sent_tokenize as split_sentences

from typing import IO
from typing import List


log = logging.get('text.encoder')


class Tokenizer:

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

        return {**res, **{
            'decoded': decoded
        }}


class Model:

    pass


# ---


@dataclass
class TransformContext:

    select: loader.SQLite

    fd_sentences: IO[str]
    fd_indexes: IO[str]

    h5fd: h5py.File
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

    # ---

    ctx.fd_sentences.write(
        '# Format: <ID> <NAME> <SENTENCE>\n'.encode())
    ctx.fd_indexes.write(
        '# Format: <ID>, <IDXS>, <TOKEN_TYPES>, <ATT_MASK>\n'.encode())

    # ---

    ents = part.entities
    shape = len(ents), ctx.sentences, 3, ctx.tokens

    # BERT vocabulary size is 30k -> 2**16 is around 65k
    log.info(f'creating dataset of size {shape}')
    ds = ctx.h5fd.create_dataset(part.name, shape, dtype='uint16')

    gen = ((e, ctx.dataset.id2ent[e]) for e in list(ents)[:10])
    for e, name in helper.tqdm(gen):
        result = ctx.select.by_entity(name)  # TODO replace with id selection
        texts = list(zip(*result))[1]

        # TODO remove; then do '\n'.join(...).split('\n')
        texts = map(split_sentences, texts)
        sentences = [t for sub in texts for t in sub]

        random.shuffle(sentences)
        sentences = sentences[:ctx.sentences]
        toks = ctx.tokenizer(
            sentences=sentences,
            padding='max_length',
            max_length=ctx.tokens,
            truncation=True)

        # write
        ctx.fd_sentences.write(('\n'.join(
            f'{e} {name} {s}'
            for s in toks['decoded']) + '\n').encode())

        ctx.fd_indexes.write(('\n'.join(
            f'{e}, '
            f'{_ints2str(toks["input_ids"][i])}, '
            f'{_ints2str(toks["token_type_ids"][i])} '
            f'{_ints2str(toks["attention_mask"][i])}'
            for i, _ in enumerate(sentences)) + '\n').encode())

        n = len(sentences)
        ds[e, :n, 0] = toks['input_ids']
        ds[e, :n, 1] = toks['token_type_ids']
        ds[e, :n, 2] = toks['attention_mask']


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

    """

    p_out = (
        ryn.ENV.TEXT_DIR /
        'transformed' /
        dataset.name /
        f'{model}.{sentences}'
    )

    log.info(f'creating {p_out}')
    p_out.mkdir(exist_ok=True, parents=True)

    with (p_out / 'info.txt').open(mode='w') as fd:
        fd.write('\n'.join((
            f'created: {datetime.now()}',
            f'git hash: {helper.git_hash()}',
            f'dataset: {dataset.name}',
            f'database: {pathlib.Path(database).name}',
            f'sentences: {sentences}',
            f'tokens: {tokens}',
            f'model: {model}')) + '\n')

    # --

    with contextlib.ExitStack() as stack:

        ctx = TransformContext(
            fd_sentences=stack.enter_context(
                gzip.open(str(p_out / 'sentences.txt.gz'), mode='wb')),
            fd_indexes=stack.enter_context(
                gzip.open(str(p_out / 'indexes.txt.gz'), mode='wb')),
            select=stack.enter_context(
                loader.SQLite(database=database)),
            h5fd=stack.enter_context(
                h5py.File(str(p_out / 'idxs.h5'), mode='w')),

            tokenizer=Tokenizer(model=model),
            dataset=dataset,
            sentences=sentences,
            tokens=tokens,
        )

        _transform_split(ctx, dataset.cw_train)


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


def embed():
    pass
