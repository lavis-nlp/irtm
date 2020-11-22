# -*- coding: utf-8 -*-

import ryn

import os
import pathlib
import logging
import configparser

from logging.config import fileConfig


ENV_RYN_LOG_CONF = 'RYN_LOG_CONF'
ENV_RYN_LOG_OUT = 'RYN_LOG_OUT'


def _init_logging():

    # probe environment for logging configuration:
    #   1. if conf/logging.conf exists use this
    #   2. if RYN_LOG_CONF is set as environment variable use its value
    #      as path to logging configuration

    fconf = None

    if ENV_RYN_LOG_CONF in os.environ:
        fconf = str(os.environ[ENV_RYN_LOG_CONF])

    else:
        path = pathlib.Path(ryn.ENV.CONF_DIR / 'logging.conf')
        if path.is_file():
            cp = configparser.ConfigParser()
            cp.read(path)

            opt = cp['handler_fileHandler']
            (fname, ) = eval(opt['args'])

            if ENV_RYN_LOG_OUT in os.environ:
                fname = pathlib.Path(os.environ[ENV_RYN_LOG_OUT])
            else:
                fname = ryn.ENV.ROOT_DIR / fname

            fname.parent.mkdir(exist_ok=True, parents=True)
            fname.touch(exist_ok=True)
            opt['args'] = repr((str(fname), ))

            fconf = cp

    if fconf is not None:
        fileConfig(cp)

# ---


def get(name: str) -> logging.Logger:
    return logging.getLogger(f'ryn.{name}')


# ---


# be nice if used as a library - do not log to stderr as default
log = logging.getLogger('ryn')
log.addHandler(logging.NullHandler())
_init_logging()

log = get('common')
log.info('-' * 80)
log.info('initialized logging')
