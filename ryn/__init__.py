# -*- coding: utf-8 -*-

import os
import sys
import pathlib

from ryn.common import logging


log = logging.get('ryn')

_root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
_root_path = pathlib.Path(_root_dir)


_DATA_DIR = 'data'


class ENV:
    ROOT_DIR:  pathlib.Path = _root_path

    LIB_DIR:   pathlib.Path = _root_path / 'lib'
    CONF_DIR:  pathlib.Path = _root_path / 'conf'

    DATA_DIR:  pathlib.Path = _root_path / _DATA_DIR
    CACHE_DIR: pathlib.Path = _root_path / _DATA_DIR / 'cache'
    EXP_DIR:   pathlib.Path = _root_path / _DATA_DIR / 'experiments'


log.info(f'! initialized path to ryn: {ENV.ROOT_DIR}')


class RynError(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)
