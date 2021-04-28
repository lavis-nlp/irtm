# -*- coding: utf-8 -*-

import irtm

import os
import pathlib
import logging
import configparser

from logging.config import fileConfig


ENV_IRTM_LOG_CONF = 'IRTM_LOG_CONF'
ENV_IRTM_LOG_OUT = 'IRTM_LOG_OUT'


def _init_logging():

    # probe environment for logging configuration:
    #   1. if conf/logging.conf exists use this
    #   2. if IRTM_LOG_CONF is set as environment variable use its value
    #      as path to logging configuration

    fconf = None

    if ENV_IRTM_LOG_CONF in os.environ:
        fconf = str(os.environ[ENV_IRTM_LOG_CONF])

    else:
        path = pathlib.Path(irtm.ENV.CONF_DIR / 'logging.conf')
        if path.is_file():
            cp = configparser.ConfigParser()
            cp.read(path)

            opt = cp['handler_fileHandler']
            (fname, ) = eval(opt['args'])

            if ENV_IRTM_LOG_OUT in os.environ:
                fname = pathlib.Path(os.environ[ENV_IRTM_LOG_OUT])
            else:
                fname = irtm.ENV.ROOT_DIR / fname

            fname.parent.mkdir(exist_ok=True, parents=True)
            fname.touch(exist_ok=True)
            opt['args'] = repr((str(fname), ))

            fconf = cp

    if fconf is not None:
        fileConfig(cp)

# ---


def get(name: str) -> logging.Logger:
    return logging.getLogger(f'irtm.{name}')


# ---


# be nice if used as a library - do not log to stderr as default
log = logging.getLogger('irtm')
log.addHandler(logging.NullHandler())
_init_logging()

log = get('common')
log.info('-' * 80)
log.info('initialized logging')
