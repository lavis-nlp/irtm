# -*- coding: utf-8 -*-


import ryn

from ryn.text import loader
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import pathlib
import contextlib

from dataclasses import dataclass

import h5py
import transformers as tf

from typing import List


log = logging.get('text.encoder')


class Tokenizer:

    @helper.notnone
    def __init__(self, model: str = None):
        cache_dir = str(ryn.ENV.CACHE_DIR / 'lib.transformers')
        self._tok = tf.BertTokenizer.from_pretrained(
            model, cache_dir=cache_dir)

    @helper.notnone
    def __call__(self, sentences: List[str] = None):

        # sets 'input_ids', 'token_type_ids', 'attention_mask'
        res = self._tok(sentences, padding=True, truncation=True)
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
    h5fd: h5py.File
    tokenizer: Tokenizer

    contexts: int
    tokens: int


def _transform_split(ctx: TransformContext, part: split.Part):
    log.info(f'transforming split {part.name}')

    ents = part.entities
    shape = len(ents), ctx.contexts, ctx.tokens

    # BERT vocabulary size is 30k -> 2**16 is around 65k
    ctx.h5fd.create_dataset(part.name, shape, dtype='uint16')

    for e in ents:
        result = ctx.select.by_entity(e)
        texts = list(zip(*result))[1]
        toks = ctx.tokenizer(texts)


@helper.notnone
def transform(
        *,
        dataset: split.Dataset = None,
        database: str = None,
        contexts: int = None,
        tokens: int = None,
        model: str = None):
    """

    Tokenize and map the text for a dataset

    """

    p_db = pathlib.Path(database)
    p_out = ryn.ENV.TEXT_DIR / 'transformed' / dataset.name / p_db.name

    log.info(f'creating {p_out}')
    p_out.mkdir(exist_ok=True, parents=True)

    # --

    with contextlib.ExitStack() as stack:

        ctx = TransformContext(
            select=stack.enter_context(
                loader.SQLite(database=database)),
            h5fd=stack.enter_context(
                h5py.File(str(p_out / 'idxs.h5'), mode='w')),
            tokenizer=Tokenizer(model=model),
            contexts=contexts,
            tokens=tokens,
        )

        _transform_split(ctx, dataset.cw_train)


def _transform_from_args(args):
    assert args.dataset, 'provide --dataset'
    assert args.database, 'provide --database'
    assert args.contexts, 'provide --contexts'
    assert args.tokens, 'provide --tokens'
    assert args.model, 'provide --model'

    dataset = split.Dataset.load(args.dataset)
    transform(
        dataset=dataset,
        database=args.database,
        contexts=args.contexts,
        tokens=args.tokens,
        model=args.model, )


def embed():
    pass
