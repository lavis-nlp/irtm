# -*- coding: utf-8 -*-

from ryn import RynError
from ryn.common import logging

import os
import pathlib
import inspect

from datetime import datetime


log = logging.get('common.helper')


# --- DECORATOR


def notnone(fn):
    spec = inspect.getfullargspec(fn)
    kwarg_names = spec.args[-len(spec.defaults):]

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
