# -*- coding: utf-8 -*-

from irtm import IRTMError

import git
import torch
import numpy as np
import numpy.random
from tqdm import tqdm as _tqdm

import os
import pickle
import random
import pathlib
import logging

from datetime import datetime
from functools import wraps
from functools import partial

from typing import List
from typing import Union
from typing import Callable


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


def timed(fn, name="unknown"):
    @wraps(timed)
    def _proxy(*args, **kwargs):

        ts = datetime.now()
        ret = fn(*args, **kwargs)
        delta = datetime.now() - ts

        log.info(f"execution of {fn.__qualname__} took {delta}")

        return ret

    return _proxy


class Cache:
    @property
    def fn(self) -> Callable:
        return self._fn

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def invalid(self) -> bool:
        return self._invalid

    def invalidate(self) -> None:
        self._invalid = True

    def __init__(self, filename: str, fn: Callable):

        self._fn = fn
        self._filename = filename
        self._invalid = False

    def __call__(self, *args, path: Union[str, pathlib.Path], **kwargs):
        path = pathlib.Path(path)
        cache = path / self.filename.format(**kwargs)
        name = f"{path.name}/{cache.name}"

        if not self.invalid and cache.is_file():
            log.info(f"loading from cache: {name}")
            with cache.open(mode="rb") as fd:
                return pickle.load(fd)

        if self.invalid:
            log.info(f"! invalidating {name}")

        # ---

        log.info(f"cache miss for {name}")
        obj = self.fn(*args, path=path, **kwargs)

        # ---

        with cache.open(mode="wb") as fd:
            log.info(f"writing cache file: {cache}")
            pickle.dump(obj, fd)

        return obj


def cached(filename: str):
    return lambda fn: Cache(filename=filename, fn=fn)


# --- UTILITY


def path(
    name: Union[str, pathlib.Path],
    create: bool = False,
    exists: bool = False,
    is_file: bool = False,
    message: str = None,
) -> pathlib.Path:
    # TODO describe message (see kgc.config)
    path = pathlib.Path(name)

    if (exists or is_file) and not path.exists():
        raise IRTMError(f"{path} does not exist")

    if is_file and not path.is_file():
        raise IRTMError(f"{path} exists but is not a file")

    if create:
        path.mkdir(exist_ok=True, parents=True)

    if message:
        path_abbrv = f"{path.parent.name}/{path.name}"
        log.info(message.format(path=path, path_abbrv=path_abbrv))

    return path


def path_rotate(current: Union[str, pathlib.Path]):
    """

    Rotates a file

    Given a file "foo.tar", rotating it will produce "foo.1.tar".
    If "foo.1.tar" already exists then "foo.1.tar" -> "foo.2.tar".
    And so on. Also works for directories.

    """
    current = path(current, message="rotating {path_abbrv}")

    def _new(
        p: pathlib.Path,
        n: int = None,
        suffixes: List[str] = None,
    ):
        name = p.name.split(".")[0]  # .stem returns foo.tar for foo.tar.gz
        return p.parent / "".join([name, "." + str(n)] + suffixes)

    def _rotate(p: pathlib.Path):
        if p.exists():
            n, *suffixes = p.suffixes
            new = _new(p, n=int(n[1:]) + 1, suffixes=suffixes)
            _rotate(new)
            p.rename(new)

    if current.exists():
        new = _new(current, n=1, suffixes=current.suffixes)
        _rotate(new)
        current.rename(new)


def seed(seed: int) -> np.random.Generator:
    log.info(f"! setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    return np.random.default_rng(seed)


def git_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    # dirty = '-dirty' if repo.is_dirty else ''
    return str(repo.head.object.hexsha)


def notebook():
    # %load_ext autoreload
    # %autoreload 2
    cwd = pathlib.Path.cwd()

    # TODO no longer necessary since introducing irtm.ENV?
    if cwd.name != "irtm":
        print("changing directory")
        os.chdir(cwd.parent)

    logger = logging.logging.getLogger()
    logger.setLevel(logging.logging.INFO)
