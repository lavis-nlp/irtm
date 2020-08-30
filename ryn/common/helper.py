# -*- coding: utf-8 -*-

from ryn import RynError
from ryn.common import logging

import os
import pathlib
import inspect

import git
from tqdm import tqdm as _tqdm

from datetime import datetime
from functools import partial


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

    def _proxy(*args, **kwargs):
        for argname in kwarg_names:
            if kwargs.get(argname) is None:
                msg = f'argument {argname} for {fn} must not be None'
                raise RynError(msg)

        return fn(*args, **kwargs)

    return _proxy


def timed(fn, name='unknown'):
    def _proxy(*args, **kwargs):

        ts = datetime.now()
        ret = fn(*args, **kwargs)
        delta = datetime.now() - ts

        log.info(f'execution of {fn.__qualname__} took {delta}')

        return ret

    return _proxy


# --- UTILITY


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


def git_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    # dirty = '-dirty' if repo.is_dirty else ''
    return str(repo.head.object.hexsha)
