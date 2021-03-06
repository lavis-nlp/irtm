# -*- coding: utf-8 -*-
# fmt: off

import pathlib


__version__ = '0.3'


_root_path = pathlib.Path(__file__).parent.parent
_DATA_DIR = 'data'


class ENV:

    ROOT_DIR:  pathlib.Path = _root_path

    LIB_DIR:   pathlib.Path = _root_path / 'lib'
    CONF_DIR:  pathlib.Path = _root_path / 'conf'
    SRC_DIR:   pathlib.Path = _root_path / 'irtm'

    # data heap

    DATA_DIR:    pathlib.Path = _root_path / _DATA_DIR
    KGC_DIR:     pathlib.Path = _root_path / _DATA_DIR / 'kgc'
    TEXT_DIR:    pathlib.Path = _root_path / _DATA_DIR / 'text'
    CACHE_DIR:   pathlib.Path = _root_path / _DATA_DIR / 'cache'
    SPLIT_DIR:   pathlib.Path = _root_path / _DATA_DIR / 'split'
    ARCHIVE_DIR: pathlib.Path = _root_path / _DATA_DIR / 'archive'


class IRTMError(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)
