import ryn

from ryn.text import loader as ryn_loader
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import gzip
import json
import random
import pathlib
import traceback
import contextlib
import multiprocessing as mp

from datetime import datetime
from functools import partial
from functools import lru_cache
from dataclasses import replace
from dataclasses import dataclass
from collections import Counter

import transformers as tf

from typing import IO
from typing import Any
from typing import List
from typing import Dict
from typing import Union
from typing import Optional


log = logging.get("text.prep")


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
    shuffle: bool = True,
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
                # mention in sentence,
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
    if shuffle:
        random.shuffle(sentences)

    return tuple(sentences[:amount])


def _transform_split(
    wid: int,
    ctx: TransformContext,
    part: split.Part,
    masked: bool = False,
    marked: bool = False,
    shuffle: bool = True,
):

    log.info(
        f"! transforming split {part.name} "
        f"({masked=}, {marked=}) {shuffle=}"
    )
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
            shuffle=shuffle,
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
    shuffle: bool = True,
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
    db_name = ryn_loader.LOADER[loader].db_name(**loader_args)

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
            shuffle=shuffle,
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

        kwargs = dict(masked=masked, marked=marked, shuffle=shuffle)
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
