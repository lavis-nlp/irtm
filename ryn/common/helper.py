# -*- coding: utf-8 -*-

from ryn import RynError
from ryn.common import logging

import os
import pickle
import random
import pathlib
import inspect

import git
import torch
import numpy as np
import numpy.random
from tqdm import tqdm as _tqdm

from datetime import datetime
from functools import wraps
from functools import partial

from typing import Union
from typing import Callable


log = logging.get('common.helper')
tqdm = partial(_tqdm, ncols=80)


# --- DECORATOR


def notnone(fn):

    # def f(a, b)
    # spec: args=['a', 'b'], defaults=None, kwonlyargs=[]
    #
    # def f(a, b, foo=None, bar=None)
    # spec: args=['a', 'b', 'foo', 'bar'], defaults=(None, None), kwonlyargs=[]
    #
    # def f(foo=None, bar=None)
    # spec: args=['foo', 'bar'], defaults=(None, None), kwonlyargs=[]
    #
    # def f(*, foo=None, bar=None)
    # spec: args=[], defaults=None, kwonlyargs=['foo', 'bar']

    spec = inspect.getfullargspec(fn)

    try:
        kwarg_names = spec.args[-len(spec.defaults):]

    # spec.defaults is None for kwonly functions
    except TypeError:
        kwarg_names = spec.kwonlyargs

    @wraps(fn)
    def _proxy(*args, **kwargs):
        for argname in kwarg_names:
            if kwargs.get(argname) is None:
                msg = f'argument {argname} for {fn} must not be None'
                raise RynError(msg)

        return fn(*args, **kwargs)

    return _proxy


def timed(fn, name='unknown'):
    @wraps(timed)
    def _proxy(*args, **kwargs):

        ts = datetime.now()
        ret = fn(*args, **kwargs)
        delta = datetime.now() - ts

        log.info(f'execution of {fn.__qualname__} took {delta}')

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

    @notnone
    def __init__(
            self,
            filename: str = None,
            fn: Callable = None):

        self._fn = fn
        self._filename = filename
        self._invalid = False

    def __call__(self, *args, path: Union[str, pathlib.Path], **kwargs):
        cache = pathlib.Path(path) / self.filename
        name = f'{path.name}/{self.filename}'

        if not self.invalid and cache.is_file():
            log.info(f'loading from cache: {cache}')
            with cache.open(mode='rb') as fd:
                return pickle.load(fd)

        if self.invalid:
            log.info(f'! invalidating {name}')

        # ---

        log.info(f'cache miss for {name}')
        obj = self.fn(*args, path=path, **kwargs)

        # ---

        with cache.open(mode='wb') as fd:
            log.info(f'writing cache file: {cache}')
            pickle.dump(obj, fd)

        return obj


def cached(filename: str):
    return lambda fn: Cache(filename=filename, fn=fn)


# --- UTILITY


def seed(seed: int) -> np.random.Generator:
    log.info(f'! setting seed to {seed}')
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

    # TODO no longer necessary since introducing ryn.ENV?
    if cwd.name != 'ryn':
        print('changing directory')
        os.chdir(cwd.parent)

    logger = logging.logging.getLogger()
    logger.setLevel(logging.logging.INFO)
