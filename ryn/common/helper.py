# -*- coding: utf-8 -*-

import ryn
from ryn import RynError
from ryn.common import logging

import git

import pickle
import pathlib
import inspect
from datetime import datetime
from dataclasses import dataclass
from collections import OrderedDict

from typing import Any
from typing import Callable


log = logging.get('common.helper')


# --- DECORATOR


def notnone(f):
    spec = inspect.getfullargspec(f)
    kwarg_names = spec.args[-len(spec.defaults):]

    def _proxy(*args, **kwargs):
        for argname in kwarg_names:
            if kwargs.get(argname) is None:
                msg = f'argument {argname} for {f} must not be None'
                raise RynError(msg)

        return f(*args, **kwargs)

    return _proxy


def timed(f, name='unknown'):
    def _proxy(*args, **kwargs):

        ts = datetime.now()
        ret = f(*args, **kwargs)
        delta = datetime.now() - ts

        log.info(f'execution of {f.__qualname__} took {delta}')

        return ret

    return _proxy


# --- UTILITY


def relpath(path: pathlib.Path):
    return path.relative_to(ryn.ENV.ROOT_DIR)


def git_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    return sha


# --- FILE CACHE


@dataclass
class _Cache:
    """

    see common.file_cache

    """

    # ---

    fn: Callable
    fname: str
    suffix: str
    maxsize: int

    # ---

    @property
    def missed(self) -> bool:
        return self._missed

    @property
    def p_cache(self) -> pathlib.Path:
        return self._p_cache

    @property
    def cache(self) -> OrderedDict:  # pathlib.Path -> None
        return self._cache

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def overflows(self) -> int:
        return self._overflows

    # ---

    def __post_init__(self):
        self._cache = None

    # ---

    def _lazy_init(self):
        cache = None

        p_cache = pathlib.Path(self.fname)
        p_cache.parent.mkdir(parents=True, exist_ok=True)

        if p_cache.exists():
            log.info(f'loading cache from {p_cache.name}')
            with p_cache.open(mode='rb') as fd:
                cache = pickle.load(fd)

        else:
            log.info('create new cache')
            cache = OrderedDict()

        assert cache is not None

        self._p_cache = p_cache
        self._cache = cache
        self._data = {}

        self._hits = 0
        self._misses = 0
        self._overflows = 0

    def _handle_overflow(self):
        if len(self.cache) != self.maxsize:
            return

        p, _ = self.cache.popitem(last=False)
        log.info(f'cache: removing {p} from cache')
        if p.exists():
            p.unlink()

        self._overflows += 1

    def _load_data(self, p):
        if p not in self._data:
            log.info(f'cache: loading data from {p.name.split("=")[1]}')
            if not p.exists():
                log.error('cache: file vanished, invalidating entry')
                raise KeyError()

            with p.open(mode='rb') as fd:
                self._data[p] = pickle.load(fd)

    def _write_cache(self):
        with self.p_cache.open(mode='wb') as fd:
            log.info(f'cache writing {self.p_cache.name}')
            pickle.dump(self.cache, fd)

    def __call__(self, *args, **kwargs) -> Any:
        if self.cache is None:
            self._lazy_init()

        s = self.suffix.format(*args, **kwargs).replace('/', '_')
        p = pathlib.Path(f'{self.p_cache}={s}')

        self._handle_overflow()
        assert len(self.cache) < self.maxsize

        try:
            del self.cache[p]  # raises
            self._load_data(p)

            self._hits += 1
            self._missed = False

        except KeyError:
            data = self.fn(*args, **kwargs)
            self._cache[p] = None
            self._data[p] = data

            self._misses += 1
            self._missed = True

            log.info(f'cache: writing {p.name.split("=")[1]}')
            with p.open(mode='wb') as fd:
                pickle.dump(data, fd)

        self.cache[p] = None
        self._write_cache()
        return self._data[p]


def file_cache(fname: str = None, suffix: str = None, maxsize: int = 128):
    """

    Create a LRU cache with persistence

    Note that this cache is used for relatively large, hard to
    compute items. It is not suitable for many small items -
    use the non-persistent functools.lru_cache instead.

    Since 3.3, hashes are salted and thus no longer consistent
    between independent runs of the program. As such, approaches
    like functools._make_key cannot be used. This file_cache
    uses the hashlib to create consistent hashes from strings.
    This cannot be done automatically, as the representation of
    many objects depends on their id() (which also changes between)
    runs. This is the reason for introducing the suffix argument.

    Parameters
    ----------

    fname : str
      File to store the cache in (parent directories are created)

    suffix : str
      Format to create a consistent hash from

    maxsize : int
      Entries are deleted if their number exceeds this threshold


    Example
    -------

    @file_cache(fname=str(ryn.ENV.CACHE_DIR / 'test.cache'),
                suffix='{}.{y}',
                maxsize=3)
    def test(x, y=10):
        return x + y

    assert test(1) == 11
    assert test(2, y=3) == 5
    assert test(2, y=3) == 5
    assert test(3, y=3) == 6
    assert test(3, y=3) == 6
    assert test(4, y=4) == 8

    print(test.hits)        # -> 2
    print(test.misses)      # -> 4
    print(test.overflows)   # -> 2

    The next time this function is initialized (e.g. when rerunning
    the program) the test.cache_overflows == 4 (because the the first
    invocations invalidate the current entries)

    """
    assert fname is not None
    assert suffix is not None

    kwargs = dict(fname=fname, suffix=suffix, maxsize=maxsize)
    return lambda fn: _Cache(fn=fn, **kwargs)
