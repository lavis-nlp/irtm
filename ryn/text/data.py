# -*- coding: utf-8 -*-

import ryn

from ryn.kgc import keen
from ryn.text import loader
from ryn.text.config import Config
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import re
import gzip
import json
import random
import pathlib
import textwrap
import contextlib
import multiprocessing as mp

from itertools import count
from datetime import datetime
from functools import partial
from functools import lru_cache
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.utils.data as torch_data
from torch.nn.utils.rnn import pad_sequence
from pykeen import triples as keen_triples
import transformers as tf

from typing import IO
from typing import Set
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple

log = logging.get('text.data')


SEP = ' | '


class Tokenizer:

    TOK_MENTION_START = '[MENTION_START]'
    TOK_MENTION_END = '[MENTION_END]'

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
            model,
            cache_dir=cache_dir,
            additional_special_tokens=[
                Tokenizer.TOK_MENTION_START,
                Tokenizer.TOK_MENTION_END,
            ]
        )

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


# this matches masked sequences of the upstream sqlite context dbs
RE_MASKS = re.compile('#+')
MARKED = (
    Tokenizer.TOK_MENTION_START +
    ' {mention} ' +
    Tokenizer.TOK_MENTION_END
)


def _transform_result(
        result,
        *,
        e: int = None,
        amount: int = None,
        masked: bool = None,
        marked: bool = None,
        tokenizer: Tokenizer = None,
):

    entities, mentions, blobs, blobs_masked = zip(*result)
    assert all(e == db_entity for db_entity in entities)

    def _flatten(nested):
        return [sent for blob in nested for sent in blob.split('\n')]

    sentences = _flatten(blobs_masked) if masked else _flatten(blobs)

    if masked:
        assert not marked
        sentences = [
            RE_MASKS.sub(tokenizer.base.mask_token, s)
            for s in sentences]

    if marked:
        assert not masked
        sentences = [
            s.replace(mention, MARKED.format(mention=mention))
            for s, mention in zip(sentences, mentions)
        ]

    random.shuffle(sentences)

    return tuple(
        # remove unnecessary whitespace
        ' '.join(sentence.split()) for sentence in
        sentences[:amount])


def _transform_split(
        wid: int,
        ctx: TransformContext,
        part: split.Part,
        masked: bool = False,
        marked: bool = False):

    log.info(f'! transforming split {part.name} '
             f'({masked=}, {marked=})')

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
        f'# Format: <ID>{SEP}<NAME>{SEP}<TRIPLES>\n'.encode())

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
        leave=True,
        unit=' entities',)

    no_context_triples = set()
    for i, e, name in bar(gen):
        result = ctx.select.by_entity(e)

        def _no_ctx():
            nonlocal no_context_triples

            triples = part.g.find(heads={e}, tails={e})
            count = len(triples)

            no_context_triples |= triples
            _write(ctx.fd_nocontext, f'{e}{SEP}{name}{SEP}{count}')

            msg = f'! no context for {e}: {name} ({count} triples)'
            log.info(msg)

        if not result:
            _no_ctx()
            continue

        # both clear and masked
        sentences = _transform_result(
            result,
            e=e,
            amount=ctx.sentences,
            masked=masked,
            marked=marked,
            tokenizer=ctx.tokenizer,
        )

        if not sentences:
            _no_ctx()
            log.error('no contexts after transformation')
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

        convert_ids_to_tokens = ctx.tokenizer.base.convert_ids_to_tokens
        gen = zip(sentences, indexes)
        for sentence, idx_list in gen:
            tokens = ' '.join(convert_ids_to_tokens(idx_list))
            idxstr = ' '.join(map(str, idx_list))

            if not all((sentence, tokens, idxstr)):
                log.error(f'skipping empty sentence of {e} ({name})')
                continue

            _write(ctx.fd_sentences,
                   f'{e}{SEP}{name}{SEP}{sentence}')
            _write(ctx.fd_tokens,
                   f'{e}{SEP}{name}{SEP}{tokens}')
            _write(ctx.fd_indexes,
                   f'{e}{SEP}{idxstr}')

    log.info(f'finished processing {part.name}')
    if len(no_context_triples):
        log.error(
            f'{part.name}: {len(no_context_triples)}'
            f'/{len(part.triples)} triples without context')


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
    wid, split, args, kwargs = packed

    with contextlib.ExitStack() as stack:
        log.info(f'! dispatching worker #{wid} for {split}')

        part = args.dataset[split]
        ctx = _transform_get_ctx(stack, split, args)

        _transform_split(wid, ctx, part, **kwargs)


def transform(
        *,
        dataset: split.Dataset = None,
        database: str = None,
        sentences: int = None,
        tokens: int = None,
        model: str = None,
        masked: bool = None,
        marked: bool = None,
        # optional
        suffix: str = None,
):
    """

    Tokenize and map the text for a dataset

    The produced files can be read by text.model.Data

    """

    if masked and marked:
        raise ryn.RynError('both masking and marking does not make sense')

    # cannot save that to a csv
    _conflicts = set(name for name in dataset.id2ent.values() if SEP in name)
    if _conflicts:
        raise ryn.RynError(f'entities contain "{SEP}": {_conflicts}')

    # ---

    if masked:
        _mode = 'masked'
    elif marked:
        _mode = 'marked'
    else:
        _mode = 'clean'

    name = f'{model}.{sentences}.{tokens}.{_mode}'
    if suffix:
        name = f'{name}-{suffix}'

    ds_name = dataset.name
    db_name = pathlib.Path(database).name

    p_out = ryn.ENV.TEXT_DIR / 'data' / ds_name / db_name / name
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

    with mp.Pool() as pool:
        args = WorkerArgs(
            p_out=p_out,
            dataset=dataset,
            database=database,
            sentences=sentences,
            tokens=tokens,
            model=model,
        )

        kwargs = dict(masked=masked, marked=marked)
        pool.map(_transform_worker, [
            (1, 'cw.train', args, kwargs),
            (2, 'ow.valid', args, kwargs),
            (3, 'ow.test', args, kwargs),
        ])


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

        log.info(f'obtained data for {len(id2ent)} distinct entities')

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

    ratio: int
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
    @helper.cached('.cached.text.data.dataset.pkl')
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
        test = Part.load(name='ow.test', path=path)
        # inductive, test = inductive.split_by_entity(ratio=ratio)

        self = Dataset(
            ratio=ratio,

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

        log.info(f'obtained {self.name}')
        return self


# --- for mapper training


class TorchDataset(torch_data.Dataset):

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._flat[idx]

    @helper.notnone
    def __init__(self, *, part: Part = None):
        super().__init__()

        self._flat = [
            (torch.Tensor(idxs).to(dtype=torch.long), e)
            for e, idx_lists in part.id2idxs.items()
            for idxs in idx_lists]

        self._max_len = max(len(idxs) for idxs, _ in self._flat)
        log.info('initialized Data: '
                 f'samples={len(self._flat)}, '
                 f'max sequence length: {self._max_len}')

    @property
    def collator(self):
        # TODO use trainer callback instead
        max_len = self._max_len

        def _collate_fn(batch: List[Tuple]):
            ents, idxs = zip(*batch)

            padded = pad_sequence(idxs, batch_first=True)
            shape = padded.shape[0], max_len
            bowl = torch.zeros(shape).to(dtype=torch.long)
            bowl[:, :padded.shape[1]] = padded

            return bowl, ents

        return _collate_fn

    @staticmethod
    def collate_fn(batch: List[Tuple]):
        idxs, ents = zip(*batch)
        return pad_sequence(idxs, batch_first=True), ents


@dataclass
class Models:

    @property
    def kgc_model_name(self) -> str:
        return self.kgc_model.config.model.cls.lower()

    kgc_model: keen.Model
    text_encoder: tf.BertModel

    @classmethod
    @helper.notnone
    def load(
            K, *,
            config: Config = None, ):

        text_encoder = tf.BertModel.from_pretrained(
            config.text_encoder,
            cache_dir=ryn.ENV.CACHE_DIR / 'lib.transformers')

        kgc_model = keen.Model.load(
            config.kgc_model,
            split_dataset=config.split_dataset)

        return K(
            text_encoder=text_encoder,
            kgc_model=kgc_model,
        )


@dataclass
class Triples:

    entities: Set[int]
    factory: keen_triples.TriplesFactory


@dataclass
class Datasets:

    @property
    def text_encoder(self):
        return self.text.model.lower()

    text: Dataset
    keen: keen.Dataset
    split: split.Dataset

    # mapper training
    train: torch_data.DataLoader
    valid: torch_data.DataLoader

    # kgc for mapper validation
    ryn2keen: Dict[int, int]
    inductive: Triples
    transductive: Triples

    @staticmethod
    @helper.notnone
    def _create_triples(
            *,
            config: Config = None,
            entities: Set[int] = None,
            text_part: Part = None,
            split_dataset: split.Dataset = None,
            split_part: split.Part = None,
            **kwargs,  # e2id and r2id
    ):
        # kgc
        triples = keen.triples_to_ndarray(split_dataset.g, split_part.triples)
        factory = keen_triples.TriplesFactory(triples=triples, **kwargs)

        return Triples(
            entities=entities,
            factory=factory,
        )

    @staticmethod
    @helper.notnone
    def _create_triples_factories(
            *,
            config: Config = None,
            text_dataset: Dataset = None,
            keen_dataset: keen.Dataset = None,
            split_dataset: split.Dataset = None,
    ):
        # create triple factories usable by pykeen
        # re-use original id-mapping and extend this with own ids
        relation_to_id = keen_dataset.relation_to_id
        entity_to_id = keen_dataset.entity_to_id
        split_part_ow = split_dataset.ow_valid

        # add owe entities to the entity mapping
        entity_to_id.update({
            keen.e2s(split_dataset.g, e): idx
            for e, idx in zip(split_part_ow.owe, count(len(entity_to_id)))
        })

        log.info(f'added {len(split_part_ow.owe)} ow entities to mapping')

        transductive = Datasets._create_triples(
            config=config,
            text_part=text_dataset.train | text_dataset.transductive,
            split_dataset=split_dataset,
            split_part=split_dataset.cw_valid,
            entities=split_dataset.cw_train.owe,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        inductive = Datasets._create_triples(
            config=config,
            text_part=text_dataset.inductive,
            split_dataset=split_dataset,
            split_part=split_part_ow,
            entities=split_part_ow.owe,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        log.info('created transductive/inductive triples factories')

        ryn2keen = {}
        for factory in (transductive.factory, inductive.factory):
            ryn2keen.update({
                int(name.split(':', maxsplit=1)[0]): keen_id
                for name, keen_id in factory.entity_to_id.items()
            })

        log.info(f'initialized ryn2keen mapping with {len(ryn2keen)} ids')
        return transductive, inductive, ryn2keen

    @staticmethod
    @helper.notnone
    def _create_dataloader(
            *,
            config: Config = None,
            text_dataset: Dataset = None
    ):

        # training and validation operate on reference embeddings
        # of the kgc model and thus no inductive part can be used here

        training_set = TorchDataset(part=text_dataset.train)
        train = torch_data.DataLoader(
            training_set,
            collate_fn=TorchDataset.collate_fn,
            **config.dataloader_train_args)

        validation_set = TorchDataset(part=text_dataset.transductive)
        valid = torch_data.DataLoader(
            validation_set,
            collate_fn=TorchDataset.collate_fn,
            **config.dataloader_valid_args)

        log.info('created train/valid dataloaders')
        return train, valid

    @classmethod
    @helper.notnone
    def load(K, config: Config = None, models: Models = None):
        log.info('loading datasets')

        keen_dataset = models.kgc_model.keen_dataset
        split_dataset = models.kgc_model.split_dataset

        text_dataset = Dataset.load(
            path=config.text_dataset,
            ratio=config.valid_split,
        )

        train, valid = Datasets._create_dataloader(
            config=config,
            text_dataset=text_dataset,
        )

        transductive, inductive, ryn2keen = Datasets._create_triples_factories(
            config=config,
            text_dataset=text_dataset,
            keen_dataset=keen_dataset,
            split_dataset=split_dataset,
        )

        return K(
            text=text_dataset,
            keen=keen_dataset,
            split=split_dataset,

            # mapper
            train=train,
            valid=valid,

            # kgc
            ryn2keen=ryn2keen,
            inductive=inductive,
            transductive=transductive,
        )
