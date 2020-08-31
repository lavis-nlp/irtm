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
    fd_nocontext: IO[str]

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

    # init text files

    ctx.fd_sentences.write(
        '# Format: <ID> <NAME> <SENTENCE>\n'.encode())
    ctx.fd_indexes.write(
        '# Format: <ID>, <IDXS>, <TOKEN_TYPES>, <ATT_MASK>\n'.encode())
    ctx.fd_nocontext.write(
        'No contexts were found for:\n'.encode())

    # init h5

    ents = part.entities
    shape = len(ents), ctx.sentences, 3, ctx.tokens

    # BERT vocabulary size is 30k -> 2**16 is around 65k
    log.info(f'creating dataset of size {shape}')
    assert len(ents) < 2**16, 'switch to uint32'

    h5_grp = ctx.h5fd.create_group(part.name)

    h5_idx = h5_grp.create_dataset('idx', shape, dtype='uint16')
    h5_len = h5_grp.create_dataset('len', shape[:2], dtype='uint16')
    h5_map = h5_grp.create_dataset('map', shape[:1], dtype='uint16')

    # iterate entities

    gen = list((i, e, ctx.dataset.id2ent[e]) for i, e in enumerate(ents))
    assert (len(gen) == shape[0]) and (shape[0] == len(ents))

    for i, e, name in helper.tqdm(gen):
        result = ctx.select.by_entity(name)  # TODO replace with id selection
        if not result:
            log.info(f'! no contexts found for {e}: {name}')
            ctx.fd_nocontext.write(f'{e} {name}\n'.encode())
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
            padding='max_length',
            max_length=ctx.tokens,
            truncation=True)

        # write clear text

        ctx.fd_sentences.write(('\n'.join(
            f'{e} {name} {s}'
            for s in toks['decoded']) + '\n').encode())

        ctx.fd_indexes.write(('\n'.join(
            f'{e}, '
            f'{_ints2str(toks["input_ids"][i])}, '
            f'{_ints2str(toks["token_type_ids"][i])} '
            f'{_ints2str(toks["attention_mask"][i])}'
            for i, _ in enumerate(sentences)) + '\n').encode())

        # write h5

        n = len(sentences)

        # entity position mapping
        h5_map[i] = e

        # vocabulary indexes
        h5_idx[i, :n, 0] = toks['input_ids']
        h5_idx[i, :n, 1] = toks['token_type_ids']
        h5_idx[i, :n, 2] = toks['attention_mask']

        # sequence lengths
        h5_len[i] = (h5_idx[i, :, 0] != 0).sum(axis=1)


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

    if p_out.exists():
        raise ryn.RynError(f'dataset already exists: {p_out}')

    log.info(f'creating {p_out}')
    p_out.mkdir(parents=True)

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

            fd_sentences=stack.enter_context(
                gopen(str(p_out / f'{split}-sentences.txt.gz'))),
            fd_indexes=stack.enter_context(
                gopen(str(p_out / f'{split}-indexes.txt.gz'))),
            fd_nocontext=stack.enter_context(
                gopen(str(p_out / f'{split}-nocontext.txt.gz'))),

            select=stack.enter_context(
                loader.SQLite(database=database)),

            h5fd=stack.enter_context(
                h5py.File(str(p_out / 'idxs.h5'), mode='a')),

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


def embed():
    pass
