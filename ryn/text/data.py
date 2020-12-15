# -*- coding: utf-8 -*-

import ryn

from ryn.kgc import keen
from ryn.text import loader as ryn_loader
from ryn.text.config import Config
from ryn.graphs import graph
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import gzip
import json
import random
import pathlib
import textwrap
import traceback
import contextlib
import multiprocessing as mp

from itertools import count
from datetime import datetime
from functools import partial
from functools import lru_cache
from dataclasses import field
from dataclasses import replace
from dataclasses import dataclass
from collections import Counter
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as torch_data
from torch.nn.utils.rnn import pad_sequence
from pykeen import triples as keen_triples
import pytorch_lightning as pl
import transformers as tf

from typing import IO
from typing import Any
from typing import Set
from typing import List
from typing import Dict
from typing import Union
from typing import Tuple
from typing import Optional


log = logging.get("text.data")


SEP = " | "


class Tokenizer:

    TOK_MENTION_START = "[MENTION_START]"
    TOK_MENTION_END = "[MENTION_END]"

    @property
    def base(self):
        return self._base

    @property
    @lru_cache
    def vocab(self) -> Dict[int, str]:
        return {v: k for k, v in self.base.vocab.items()}

    def __init__(
        self, model: str = None, path: Union[str, pathlib.Path] = None
    ):

        assert not (model and path), "cannot do both"

        if model:
            cache_dir = str(ryn.ENV.CACHE_DIR / "lib.transformers")
            self._base = tf.BertTokenizer.from_pretrained(
                model,
                cache_dir=cache_dir,
                additional_special_tokens=[
                    Tokenizer.TOK_MENTION_START,
                    Tokenizer.TOK_MENTION_END,
                ],
            )

        if path:
            self._base = tf.BertTokenizer.from_pretrained(str(path))

    def save(self, path: Union[str, pathlib.Path]):
        path = helper.path(path) / "tokenizer"
        self.base.save_pretrained(str(path))

    @classmethod
    def load(K, path: Union[str, pathlib.Path]):
        path = helper.path(path) / "tokenizer"
        return K(path=path)


# ---


@dataclass
class TransformContext:

    fd_indexes: IO[str]
    fd_sentences: IO[str]
    fd_tokens: IO[str]
    fd_nocontext: IO[str]

    tokenizer: Tokenizer

    dataset: split.Dataset
    sentences: int
    tokens: int

    select: Optional[ryn_loader.Loader] = None


# this matches masked sequences of the upstream sqlite context dbs
# MASK_TOKEN = "#"
# RE_MASKS = re.compile(MASK_TOKEN + "+")
MARKED = (
    Tokenizer.TOK_MENTION_START + " {mention} " + Tokenizer.TOK_MENTION_END
)


def _transform_result(
    result: ryn_loader.Result,
    *,
    e: int = None,
    amount: int = None,
    masked: bool = None,
    marked: bool = None,
    tokenizer: Tokenizer = None,
):

    #
    #  DISCLAIMER: you need to be very very careful when changing
    #  this code and always make sure the same contexts are produced
    #  for all modes (clean, marked, masked)!
    #

    def _map(sentence, mention):
        sentence = " ".join(sentence.strip().split())

        if masked:
            assert not marked
            # sentence = RE_MASKS.sub(tokenizer.base.mask_token, sentence)
            sentence = sentence.replace(mention, tokenizer.base.mask_token)

        if marked:
            assert not masked
            sentence = sentence.replace(
                mention, MARKED.format(mention=mention)
            )

        return sentence

    def _filter(sentence, mention):
        return all(
            (
                len(sentence) > 50,
                not sentence.startswith("File:"),
                mention in sentence,
            )
        )

    def _flatten(nested):

        # list(set(x)) is not possible because of the
        # unpredictable python hash seeds
        tuples, seen = [], set()

        for blob, mention in nested:

            # filter BEFORE mapping
            gen = (s for s in blob.split("\n") if _filter(s, mention))

            for sentence in gen:
                if sentence in seen:
                    continue

                seen.add(sentence)
                # map AFTER all checks
                mapped = _map(sentence, mention)
                tuples.append((mapped, mention))

        return tuples

    # blobs = result.blobs_masked if masked else result.blobs
    flat = _flatten(zip(result.blobs, result.mentions))

    # can not use list(set(X)) because of the python hash seed
    if not flat:
        return None

    sentences, mentions = map(list, zip(*flat))

    # make sure a seed is set!
    random.shuffle(sentences)
    return tuple(sentences[:amount])


def _transform_split(
    wid: int,
    ctx: TransformContext,
    part: split.Part,
    masked: bool = False,
    marked: bool = False,
):

    log.info(f"! transforming split {part.name} " f"({masked=}, {marked=})")
    helper.seed(ctx.dataset.cfg.seed)

    # ---

    # init text files

    ctx.fd_indexes.write(
        f"# Format: <ID>{SEP}<INDEX1> <INDEX2> ...\n".encode()
    )
    ctx.fd_sentences.write(
        f"# Format: <ID>{SEP}<NAME>{SEP}<SENTENCE>\n".encode()
    )
    ctx.fd_tokens.write(
        f"# Format: <ID>{SEP}<NAME>{SEP}<TOKEN1> <TOKEN2> ...\n".encode()
    )
    ctx.fd_nocontext.write(
        f"# Format: <ID>{SEP}<NAME>{SEP}<TRIPLES>\n".encode()
    )

    ents = list(part.owe)
    shape = len(ents), ctx.sentences, 3, ctx.tokens

    # iterate entities

    def _ints2str(lis: List[int]):
        return " ".join(map(str, lis))

    def _write(fd, s: str):
        assert "\n" not in s, s
        fd.write((s + "\n").encode())

    gen = list((i, e, ctx.dataset.id2ent[e]) for i, e in enumerate(ents))
    assert (len(gen) == shape[0]) and (shape[0] == len(ents))

    bar = partial(
        helper.tqdm,
        position=wid,
        desc=part.name,
        leave=True,
        unit=" entities",
    )

    no_context_entities = set()
    no_context_triples = set()

    for i, e, name in bar(gen):
        result = ctx.select.by_entity(e)

        def _no_ctx():
            nonlocal no_context_triples
            nonlocal no_context_entities

            triples = part.g.find(heads={e}, tails={e})
            count = len(triples)

            no_context_triples |= triples
            no_context_entities.add(e)
            _write(ctx.fd_nocontext, f"{e}{SEP}{name}{SEP}{count}")

            msg = f"! no context for {e}: {name} ({count} triples)"
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
            log.error("no contexts after transformation")
            continue

        # tokenize and map to vocabulary ids

        indexes = ctx.tokenizer.base(
            sentences, padding=False, max_length=ctx.tokens, truncation=True
        )["input_ids"]

        # write clear text
        # you cannot use 'decoded' for is_tokenized=True

        assert len(indexes) == len(
            sentences
        ), f"{len(indexes)=} != {len(sentences)=}"

        convert_ids_to_tokens = ctx.tokenizer.base.convert_ids_to_tokens
        gen = zip(sentences, indexes)
        for sentence, idx_list in gen:
            tokens = " ".join(convert_ids_to_tokens(idx_list))
            idxstr = " ".join(map(str, idx_list))

            if not all((sentence, tokens, idxstr)):
                log.error(f"skipping empty sentence of {e} ({name})")
                continue

            _write(ctx.fd_sentences, f"{e}{SEP}{name}{SEP}{sentence}")
            _write(ctx.fd_tokens, f"{e}{SEP}{name}{SEP}{tokens}")
            _write(ctx.fd_indexes, f"{e}{SEP}{idxstr}")

    log.info(f"finished processing {part.name}")
    if len(no_context_triples):
        log.error(
            f"[{part.name}] {len(no_context_entities)}"
            " entities without context"
            f" ({len(no_context_triples)}/{len(part.triples)} triples)"
        )


@dataclass
class WorkerArgs:

    model: str
    tokens: int
    sentences: int

    loader: str
    loader_args: Dict[str, Any]

    p_out: pathlib.Path
    dataset: split.Dataset


def _transform_get_ctx(stack, split: str, args: WorkerArgs, file_mode: str):
    gopen = partial(gzip.open, mode=file_mode)

    select = None
    if args.loader:
        select = stack.enter_context(
            ryn_loader.LOADER[args.loader](**args.loader_args)
        )

    ctx = TransformContext(
        fd_tokens=stack.enter_context(
            gopen(str(args.p_out / f"{split}-tokens.txt.gz"))
        ),
        fd_sentences=stack.enter_context(
            gopen(str(args.p_out / f"{split}-sentences.txt.gz"))
        ),
        fd_indexes=stack.enter_context(
            gopen(str(args.p_out / f"{split}-indexes.txt.gz"))
        ),
        fd_nocontext=stack.enter_context(
            gopen(str(args.p_out / f"{split}-nocontext.txt.gz"))
        ),
        tokenizer=Tokenizer(model=args.model),
        dataset=args.dataset,
        sentences=args.sentences,
        tokens=args.tokens,
        select=select,
    )

    return ctx


def _transform_worker(packed):
    wid, split, args, kwargs = packed

    with contextlib.ExitStack() as stack:
        log.info(f"! dispatching worker #{wid} for {split}")

        part = args.dataset[split]
        ctx = _transform_get_ctx(stack, split, args, "wb")

        _transform_split(wid, ctx, part, **kwargs)


def _transform_worker_wrapper(*args, **kwargs):
    try:
        _transform_worker(*args, **kwargs)
    except Exception as exc:
        log.error(traceback.format_exc())
        log.error(f"{exc}")
        raise exc


def transform(
    *,
    dataset: split.Dataset = None,
    sentences: int = None,
    tokens: int = None,
    model: str = None,
    masked: bool = None,
    marked: bool = None,
    loader: str = None,
    loader_args: Dict[str, Any] = None,
    # optional
    suffix: str = None,
):
    """

    Tokenize and map the text for a dataset

    The produced files can be read by text.model.Data

    """

    if loader not in ryn_loader.LOADER:
        raise ryn.RynError(f"unknown loader: '{loader}'")

    if masked and marked:
        raise ryn.RynError("both masking and marking does not make sense")

    # cannot save that to a csv
    _conflicts = set(name for name in dataset.id2ent.values() if SEP in name)
    if _conflicts:
        raise ryn.RynError(f'entities contain "{SEP}": {_conflicts}')

    # ---

    if masked:
        mode = "masked"
    elif marked:
        mode = "marked"
    else:
        mode = "clean"

    name = f"{model}.{sentences}.{tokens}.{mode}"
    if suffix:
        name = f"{name}-{suffix}"

    ds_name = dataset.name

    if loader == "sqlite":
        db_name = pathlib.Path(loader_args["database"]).name
    elif loader == "json":
        db_name = pathlib.Path(loader_args["fname"]).name
        loader_args.update(dict(id2ent=dataset.g.source.ents))

    p_out = ryn.ENV.TEXT_DIR / "data" / ds_name / db_name / name
    p_out.mkdir(exist_ok=True, parents=True)

    with (p_out / "info.json").open(mode="w") as fd:

        info = dict(
            created=datetime.now().isoformat(),
            git_hash=helper.git_hash(),
            dataset=dataset.name,
            database=db_name,
            sentences=sentences,
            tokens=tokens,
            model=model,
            mode=mode,
        )

        json.dump(info, fd, indent=2)

    Tokenizer(model=model).save(p_out)

    with mp.Pool() as pool:
        args = WorkerArgs(
            p_out=p_out,
            dataset=dataset,
            loader=loader,
            loader_args=loader_args,
            sentences=sentences,
            tokens=tokens,
            model=model,
        )

        kwargs = dict(masked=masked, marked=marked)
        pool.map(
            _transform_worker_wrapper,
            [
                (1, "cw.train", args, kwargs),
                (2, "ow.valid", args, kwargs),
                (3, "ow.test", args, kwargs),
            ],
        )


@helper.notnone
def reduce(
    text_dataset: Union[str, pathlib.Path] = None,
    sentences: int = None,
):
    path = helper.path(
        text_dataset, exists=True, message="reducing {path_abbrv}"
    )

    with (path / "info.json").open(mode="r") as fd:
        info = json.load(fd)

    model, old_sentences, tokens, mode = path.name.split(".")
    old_sentences = int(old_sentences)
    tokens = int(tokens)

    assert info["model"] == model
    assert info["sentences"] == old_sentences

    if old_sentences <= sentences:
        raise ryn.RynError("nothing to reduce")

    log.info(f"reducing from {old_sentences} to {sentences} sentences")

    name = f"{model}.{sentences}.{tokens}.{mode}"
    p_out = helper.path(
        path.parent / name, create=True, message="writing to {path_abbrv}"
    )

    # --

    tokenizer = Tokenizer.load(text_dataset)
    tokenizer.save(p_out)

    info["created"] = datetime.now().isoformat()
    info["sentences"] = sentences
    info["git_hash"] = helper.git_hash()
    with (p_out / "info.json").open(mode="w") as fd:
        json.dump(info, fd, indent=2)

    # --

    in_args = WorkerArgs(
        # copy
        model=info["model"],
        dataset=info["dataset"],
        tokens=tokens,
        sentences=old_sentences,
        p_out=path,
        # unused
        loader=None,
        loader_args=None,
    )

    out_args = replace(in_args, p_out=p_out, sentences=sentences)

    def _write(out_ctx, sentence, indexes, tokens):
        out_ctx.fd_sentences.write(sentence)
        out_ctx.fd_indexes.write(indexes)
        out_ctx.fd_tokens.write(tokens)

    for part in ("cw.train", "ow.valid", "ow.test"):
        with contextlib.ExitStack() as stack:
            in_ctx = _transform_get_ctx(stack, part, in_args, "rb")
            out_ctx = _transform_get_ctx(stack, part, out_args, "wb")

            header, *lines = zip(
                *(
                    in_ctx.fd_sentences.readlines(),
                    in_ctx.fd_indexes.readlines(),
                    in_ctx.fd_tokens.readlines(),
                )
            )

            # copy header line
            _write(out_ctx, *header)

            gen = (
                (int(sent.decode().split(SEP, maxsplit=1)[0]), [sent] + other)
                for sent, *other in lines
            )

            counts = Counter()
            for e, lines in helper.tqdm(gen, unit=" sentence"):
                if counts[e] < sentences:
                    counts[e] += 1
                    _write(out_ctx, *lines)


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

    def __or__(self, other: "Part") -> "Part":
        return Part(
            name=f"{self.name}|{other.name}",
            id2sents={**self.id2sents, **other.id2sents},
            id2toks={**self.id2toks, **other.id2toks},
            id2idxs={**self.id2idxs, **other.id2idxs},
            id2ent={**self.id2ent, **other.id2ent},
        )

    def __str__(self) -> str:
        summed = sum(len(sents) for sents in self.id2sents.values())
        avg = (summed / len(self.id2sents)) if len(self.id2sents) else 0

        return "\n".join(
            (
                f"Part: {self.name}",
                f"  total entities: {len(self.id2ent)}",
                f"  average sentences per entity: {avg:2.2f}",
            )
        )

    def check(self):
        log.info(f"running self check for {self.name}")

        assert len(self.id2sents) == len(
            self.id2ent
        ), f"{len(self.id2sents)=} != {len(self.id2ent)=}"
        assert len(self.id2sents) == len(
            self.id2toks
        ), f"{len(self.id2sents)=} != {len(self.id2toks)=}"
        assert len(self.id2sents) == len(
            self.id2idxs
        ), f"{len(self.id2sents)=} != {len(self.id2idxs)=}"

        def _deep_check(d1: Dict[int, List], d2: Dict[int, List]):
            for e in d1:
                assert len(d1[e]) == len(
                    d2[e]
                ), f"{len(d1[e])=} != {len(d2[e])=}"

        _deep_check(self.id2sents, self.id2toks)
        _deep_check(self.id2sents, self.id2idxs)

    @helper.notnone
    def split_by_entity(
        self, *, ratio: float = None, retained_entities: Set[int] = None
    ):

        n = int(len(self.id2idxs) * ratio) - len(retained_entities)
        assert (0 < n) and (n < len(self.id2idxs))

        candidates = list(set(self.id2ent) - retained_entities)
        a, b = candidates[:n], candidates[n:]

        log.info(
            f"split {self.name} by entity at "
            f"{n}/{len(self.id2idxs)} ({ratio=})"
        )

        def _dic_subset(dic, keys):
            return {k: v for k, v in dic.items() if k in keys}

        def _split_by_entities(entities):
            return dict(
                id2toks=_dic_subset(self.id2toks, entities),
                id2idxs=_dic_subset(self.id2idxs, entities),
                id2sents=_dic_subset(self.id2sents, entities),
                id2ent=_dic_subset(self.id2ent, entities),
            )

        p1 = Part(name="-entity_split_a", **_split_by_entities(a))
        p2 = Part(name="-entity_split_b", **_split_by_entities(b))

        p1.check()
        p2.check()

        return p1, p2

    @helper.notnone
    def split_text_contexts(self, *, ratio: float = None):
        log.info(f"split {self.name} text contexts ({ratio=})")

        p1 = Part(name=self.name + "-context_split_a")
        p2 = Part(name=self.name + "-context_split_b")

        def _split(lis):
            if len(lis) == 1:
                return lis, []

            n = int(len(lis) * ratio)
            return lis[:n], lis[n:]

        for e in self.id2ent:
            id2toks = _split(self.id2toks[e])
            id2idxs = _split(self.id2idxs[e])
            id2sents = _split(self.id2sents[e])

            assert len(id2toks[0]), f"{e}: {self.id2toks[e]}"

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
        log.info(f"finished split: {_n} have no validation contexts")
        return p1, p2

    # ---

    @staticmethod
    def _read(path, fname):
        with gzip.open(str(path / fname), mode="r") as fd:
            log.info(f"reading {path.name}/{fname}")
            fd.readline()  # consume head comment

            for line in map(bytes.decode, fd.readlines()):
                e_str, blob = line.split(SEP, maxsplit=1)
                yield int(e_str), blob

    @classmethod
    @helper.notnone
    def load(K, *, name: str = None, path: pathlib.Path = None):
        log.info(f"loading dataset from {path}")

        read = partial(Part._read, path)
        id2ent = {}

        # sentences
        id2sents = defaultdict(list)
        for e, blob in read(f"{name}-sentences.txt.gz"):
            e_name, sentence = blob.split(SEP, maxsplit=1)
            id2sents[e].append(sentence.strip())
            id2ent[e] = e_name

        # tokens
        id2toks = defaultdict(list)
        for e, blob in read(f"{name}-tokens.txt.gz"):
            _, tokens = blob.split(SEP, maxsplit=1)
            id2toks[e].append(tuple(tokens.split()))

        # # indexes
        id2idxs = defaultdict(list)
        for e, blob in read(f"{name}-indexes.txt.gz"):
            id2idxs[e].append(tuple(map(int, blob.split())))

        log.info(f"obtained data for {len(id2ent)} distinct entities")

        part = K(
            name=name,
            id2ent=id2ent,
            id2toks=id2toks,
            id2idxs=id2idxs,
            id2sents=id2sents,
        )

        part.check()
        return part


@dataclass
class TextDataset:
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

    # --- closed world part of ryn.split.Dataset

    # for training projections and transductive
    # knowledge graph completion
    cw_train: Part
    # validate projections for known entities
    # but unknown contexts
    cw_transductive: Optional[Part]
    # validate projections for both unknown entities
    # and unknown contexts
    cw_inductive: Optional[Part]

    # --- open world part of ryn.split.Dataset

    # for inductive knowledge graph completion
    ow_valid: Part
    # final test set for inductive kgc
    ow_test: Part

    # -- optional

    ratio: Optional[int] = None

    @property
    def name(self) -> str:
        return (
            f"{self.dataset}/{self.database}/{self.model}"
            f".{self.max_sentence_count}.{self.max_token_count}"
        )

    def __str__(self) -> str:
        buf = "\n".join(
            (
                "ryn.text.data.Dataset",
                f"{self.name}",
                f"created: {self.created}",
                f"git_hash: {self.git_hash}",
                "",
            )
        )

        cw = set(self.cw_train.id2ent)

        for name, part in (
            ("cw.train", self.cw_train),
            ("cw.transductive", self.cw_transductive),
            ("cw.inductive", self.cw_inductive),
            ("ow.valid", self.ow_valid),
            ("ow.test", self.ow_test),
        ):
            if part is None:
                buf += textwrap.indent(f"[{name.upper()}] None\n", "  ")
                continue

            ow = set(part.id2ent) - cw
            cw |= ow

            buf += (
                textwrap.indent(
                    f"[{name.upper()}] {part}\n"
                    f"  open world entities: {len(ow)}",
                    "  ",
                )
                + "\n"
            )

        return buf

    @classmethod
    @helper.notnone
    @helper.cached(".cached.text.data.dataset.{seed}.{ratio}.pkl")
    def create(
        K,
        path: Union[str, pathlib.Path],
        retained_entities: Set[int] = None,
        ratio: float = None,
        seed: int = None,
    ):

        helper.seed(seed)

        path = pathlib.Path(path)
        if not path.is_dir():
            raise ryn.RynError(f"Dataset cannot be found: {path}")

        with (path / "info.json").open(mode="r") as fd:
            info = json.load(fd)

        # create splits

        cw_train = Part.load(name="cw.train", path=path)

        # first set aside some entities to measure
        # inductive validation loss
        cw_train, cw_inductive = cw_train.split_by_entity(
            ratio=ratio, retained_entities=retained_entities
        )

        # and then split the text contexts for the remaining
        # closed world entities to measure transductive projection loss
        cw_train, cw_transductive = cw_train.split_text_contexts(ratio=ratio)

        self = TextDataset(
            ratio=ratio,
            # update
            created=datetime.now().isoformat(),
            git_hash=helper.git_hash(),
            # copy
            model=info["model"],
            dataset=info["dataset"],
            database=info["database"],
            max_sentence_count=info["sentences"],
            max_token_count=info["tokens"],
            # create
            cw_train=cw_train,
            cw_transductive=cw_transductive,
            cw_inductive=cw_inductive,
            ow_valid=Part.load(name="ow.valid", path=path),
            ow_test=Part.load(name="ow.test", path=path),
        )

        log.info(f"obtained {self}")
        return self

    @classmethod
    @helper.notnone
    @helper.cached(".cached.text.data.dataset.loaded.pkl")
    def load(
        K,
        path: Union[str, pathlib.Path] = None,
    ):
        """

        Single-sentence scenario

        """
        path = pathlib.Path(path)
        if not path.is_dir():
            raise ryn.RynError(f"Dataset cannot be found: {path}")

        with (path / "info.json").open(mode="r") as fd:
            info = json.load(fd)
        self = TextDataset(
            # update
            created=datetime.now().isoformat(),
            git_hash=helper.git_hash(),
            # copy
            model=info["model"],
            dataset=info["dataset"],
            database=info["database"],
            max_sentence_count=info["sentences"],
            max_token_count=info["tokens"],
            # load
            cw_train=Part.load(name="cw.train", path=path),
            cw_transductive=None,
            cw_inductive=None,
            ow_valid=Part.load(name="ow.valid", path=path),
            ow_test=Part.load(name="ow.test", path=path),
        )

        return self


@dataclass
class Triples:

    g: graph.Graph
    entities: Set[int]
    factory: keen_triples.TriplesFactory

    @classmethod
    def create(
        K,
        *,
        split_dataset: split.Dataset = None,
        split_part: split.Part = None,
        **kwargs,  # e2id and r2id
    ):
        # kgc
        triples = keen.triples_to_ndarray(split_dataset.g, split_part.triples)
        factory = keen_triples.TriplesFactory.from_labeled_triples(
            triples, **kwargs
        )

        return K(
            g=split_part.g,
            entities=split_part.owe,
            factory=factory,
        )


@dataclass
class KGCDataset:

    ryn2keen: Dict[int, int]
    inductive: Triples
    transductive: Triples
    test: Triples

    @classmethod
    def create(
        K,
        *,
        config: Config = None,
        keen_dataset: keen.Dataset = None,
        split_dataset: split.Dataset = None,
    ):
        # create triple factories usable by pykeen
        # re-use original id-mapping and extend this with own ids
        relation_to_id = keen_dataset.relation_to_id
        entity_to_id = keen_dataset.entity_to_id
        owe = (split_dataset.ow_valid | split_dataset.ow_test).owe

        # add owe entities to the entity mapping
        entity_to_id.update(
            {
                keen.e2s(split_dataset.g, e): idx
                for e, idx in zip(owe, count(len(entity_to_id)))
            }
        )

        log.info(f"added {len(owe)} ow entities to mapping")

        transductive = Triples.create(
            split_dataset=split_dataset,
            split_part=split_dataset.cw_train | split_dataset.cw_valid,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        inductive = Triples.create(
            split_dataset=split_dataset,
            split_part=split_dataset.ow_valid,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        test = Triples.create(
            split_dataset=split_dataset,
            split_part=split_dataset.ow_test,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

        log.info("created datasets triples factories")

        ryn2keen = {}
        for factory in (transductive.factory, inductive.factory, test.factory):
            ryn2keen.update(
                {
                    int(name.split(":", maxsplit=1)[0]): keen_id
                    for name, keen_id in factory.entity_to_id.items()
                }
            )

        log.info(f"initialized ryn2keen mapping with {len(ryn2keen)} ids")

        return K(
            ryn2keen=ryn2keen,
            transductive=transductive,
            inductive=inductive,
            test=test,
        )


# --- for mapper training


class TorchDataset(torch_data.Dataset):

    text_dataset: TextDataset
    kgc_dataset: KGCDataset

    @property
    def name(self):
        return self._name

    @property
    def max_sequence_length(self):
        return self._max_len

    @property
    def max_sequence_idx(self):
        return self._max_idx

    @property
    def degrees(self):
        return self._degrees

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._flat[idx]

    def __init__(
        self,
        *,
        name: str = None,
        part: Part = None,
        # required to obtain node degrees
        triples: Triples = None,
    ):
        assert name is not None
        assert part is not None

        super().__init__()
        self._name = name

        self._flat = [
            (torch.Tensor(idxs).to(dtype=torch.long), e)
            for e, idx_lists in part.id2idxs.items()
            for idxs in idx_lists
        ]

        lens = np.array([len(idxs) for idxs, _ in self._flat])
        self._max_idx = np.argmax(lens)
        self._max_len = lens[self.max_sequence_idx]

        log.info(
            f"initialized torch dataset {name}: samples={len(self)};"
            f" max sequence length: {self.max_sequence_length}"
        )

        if not triples:
            self._degrees = None
            return

        # note: this does not consider direction; nodes with an
        #  out_degree of 0 can still have a large in_degree
        self._degrees = torch.Tensor(
            [triples.g.nx.degree[e] for _, e in self._flat]
        )
        assert (
            len(self.degrees[self.degrees == 0]) == 0
        ), "found disconnected nodes"

        log.info(
            f"node degrees: mean={self.degrees.mean():2.2f};"
            f" std={self.degrees.std():2.2f}"
        )

    @property
    def collator(self):
        # TODO use trainer callback instead
        max_len = self._max_len

        def _collate_fn(batch: List[Tuple]):
            ents, idxs = zip(*batch)

            padded = pad_sequence(idxs, batch_first=True)
            shape = padded.shape[0], max_len
            bowl = torch.zeros(shape).to(dtype=torch.long)
            bowl[:, : padded.shape[1]] = padded

            return bowl, ents

        return _collate_fn

    @staticmethod
    def collate_fn(batch: List[Tuple]):
        idxs, ents = zip(*batch)
        return pad_sequence(idxs, batch_first=True), ents


@dataclass
class Models:

    tokenizer: Tokenizer
    kgc_model: keen.Model
    text_encoder: tf.BertModel

    @classmethod
    @helper.notnone
    def load(
        K,
        *,
        config: Config = None,
    ):

        text_encoder = None
        tokenizer = None
        text_encoder = tf.BertModel.from_pretrained(
            config.text_encoder,
            cache_dir=ryn.ENV.CACHE_DIR / "lib.transformers",
        )

        tokenizer = Tokenizer.load(config.text_dataset)

        log.info(f"resizing token embeddings to {len(tokenizer.base)}")
        text_encoder.resize_token_embeddings(len(tokenizer.base))

        # --

        kgc_model = keen.Model.load(
            config.kgc_model, split_dataset=config.split_dataset
        )

        return K(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            kgc_model=kgc_model,
        )


class DataModule(pl.LightningDataModule):
    @property
    def config(self) -> Config:
        return self._config

    @property
    def split(self) -> split.Dataset:
        return self._split

    @property
    def text(self) -> Optional[TextDataset]:
        return self._text

    @property
    def keen(self) -> Optional[keen.Dataset]:
        return self._keen

    @property
    def kgc(self) -> Optional[KGCDataset]:
        return self._kgc

    # ---
    # training utilities

    def has_geometric_validation(self) -> bool:
        return self.text.cw_inductive and self.text.cw_transductive

    def geometric_validation_kind(self, dataloader_idx: int) -> str:
        assert self.has_geometric_validation()
        return "transductive" if dataloader_idx == 0 else "inductive"

    def should_evaluate_geometric(self, dataloader_idx: Optional[int]) -> bool:
        if not self.has_geometric_validation():
            assert dataloader_idx is None
            return False

        return dataloader_idx == 0 or dataloader_idx == 1

    def should_evaluate_kgc(self, dataloader_idx: Optional[int]) -> bool:
        return dataloader_idx is None or dataloader_idx == 2

    @property
    def kgc_dataloader(self):
        dataloader = self.val_dataloader()
        if self.has_geometric_validation():
            return dataloader[2]
        return dataloader[0]

    # ---

    @helper.notnone
    def __init__(
        self,
        *,
        config: Config = None,
        keen_dataset=None,
        split_dataset=None,
    ):
        super().__init__()

        self._config = config
        self._split = split_dataset

        if self.config.split_text_dataset:
            if not self.config.valid_split:
                raise ryn.RynError(
                    "config.valid_split required if"
                    " config.split_text_dataset is set"
                )

            log.info("creating text dataset: enable geometric validation")
            self._text = TextDataset.create(
                path=self.config.text_dataset,
                retained_entities=self.split.concepts,
                ratio=self.config.valid_split,
                seed=self.split.cfg.seed,
            )
        else:
            log.info("loading text dataset: disable geometric validation")
            self._text = TextDataset.load(
                path=self.config.text_dataset,
            )

        self._keen = keen_dataset
        self._kgc = KGCDataset.create(
            config=config,
            keen_dataset=self.keen,
            split_dataset=self.split,
        )

    def prepare_data(self):
        # called once on master for multi-gpu setups
        # do not set any state here
        pass

    def setup(self, arg):

        self._train_set = TorchDataset(
            name="cw.train",
            part=self.text.cw_train,
            triples=self.kgc.transductive,
        )

        if self.has_geometric_validation():
            log.info("! dataset offers geometric validation")

            self._valid_sets = (
                TorchDataset(
                    name="cw.valid.transductive",
                    part=self.text.cw_transductive,
                ),
                TorchDataset(
                    name="cw.valid.inductive",
                    part=self.text.cw_inductive,
                ),
                TorchDataset(
                    name="ow.valid",
                    part=self.text.ow_valid,
                ),
            )
        else:
            log.info("! dataset offers no geometric validation")

            self._valid_sets = (
                TorchDataset(
                    name="ow.valid",
                    part=self.text.ow_valid,
                ),
            )

        self._test_set = TorchDataset(
            name="ow.test",
            part=self.text.ow_test,
        )

    # FOR LIGHTNING

    def train_dataloader(self) -> torch_data.DataLoader:
        sampler = None
        if self.config.sampler:
            assert self.config.sampler == "node degree"  # TODO define registry

            num_samples = self.config.sampler_args["num_samples"]
            if num_samples.startswith("x"):
                num_samples = len(self._train_set) * int(num_samples[1:])

            replacement = self.config.sampler_args["replacement"]

            sampler = torch_data.WeightedRandomSampler(
                weights=self._train_set.degrees,
                num_samples=num_samples,
                replacement=replacement,
            )

            log.info(
                f"using node degreee sampler {num_samples=} {replacement=}"
            )

        return torch_data.DataLoader(
            self._train_set,
            collate_fn=TorchDataset.collate_fn,
            sampler=sampler,
            **self.config.dataloader_train_args,
        )

    def val_dataloader(self) -> torch_data.DataLoader:
        assert self._valid_sets

        return [
            torch_data.DataLoader(
                dataset,
                collate_fn=TorchDataset.collate_fn,
                **self.config.dataloader_valid_args,
            )
            for dataset in self._valid_sets
        ]

    def test_dataloader(self) -> torch_data.DataLoader:
        # see evaluator.py
        return torch_data.DataLoader(
            self._test_set,
            collate_fn=TorchDataset.collate_fn,
            **self.config.dataloader_test_args,
        )
