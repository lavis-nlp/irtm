# -*- coding: utf-8 -*-

import pathlib

_root_path = pathlib.Path(__file__).parent.parent
_DATA_DIR = 'data'


class ENV:

    ROOT_DIR:  pathlib.Path = _root_path

    LIB_DIR:   pathlib.Path = _root_path / 'lib'
    CONF_DIR:  pathlib.Path = _root_path / 'conf'
    SRC_DIR:   pathlib.Path = _root_path / 'ryn'
    TEST_DIR:  pathlib.Path = _root_path / 'ryn' / 'tests'

    # data heap

    DATA_DIR:  pathlib.Path = _root_path / _DATA_DIR
    CACHE_DIR: pathlib.Path = _root_path / _DATA_DIR / 'cache'
    SPLIT_DIR: pathlib.Path = _root_path / _DATA_DIR / 'split'
    EMBER_DIR: pathlib.Path = _root_path / _DATA_DIR / 'ember'


class RynError(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)
