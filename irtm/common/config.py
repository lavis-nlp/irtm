# -*- coding: utf-8 -*-

from ryn import RynError
from ryn.common import helper
from ryn.common import logging

import argparse
import validate
import configobj

from dataclasses import dataclass

from typing import Callable
from typing import Generator


S_DEFAULTS = 'defaults'
S_CONSTANTS = 'constants'

log = logging.get('common.config')


@dataclass
class Config:
    """

    Defaults are overwritten by other sections.
    Constants overwrite everything.

    """

    name: str
    obj: configobj.ConfigObj

    file_name: str
    file_spec: str

    @staticmethod
    @helper.notnone
    def create(
            *,
            fconf: str = None,
            fspec: str = None) -> Generator['Config', None, None]:

        log.info(f'creating configuration generator from {fconf}')

        parser = configobj.ConfigObj(fconf)
        validator = validate.Validator()

        # FIXME: necessary?
        if S_DEFAULTS not in parser:
            raise RynError(f'you need to supply a [{S_DEFAULTS}] section')

        for section in parser:
            if section == S_DEFAULTS or section == S_CONSTANTS:
                continue

            log.info(f'reading section "{section}"')

            merged = parser[S_DEFAULTS]
            merged.merge(parser[section])

            if S_CONSTANTS in parser:
                merged.merge(parser[S_CONSTANTS])

            subparser = configobj.ConfigObj(infile=merged, configspec=fspec)
            subparser.validate(validator, preserve_errors=True)

            yield Config(
                name=section,
                obj=subparser,
                file_name=fconf,
                file_spec=fspec, )

    @staticmethod
    @helper.notnone
    def execute(
            *,
            fconf: str = None,
            fspec: str = None,
            callback: Callable[['Config'], None] = None):

        for cfg in Config.create(fconf=fconf, fspec=fspec):
            try:
                log.info(f'invoking callback for {cfg.name}')
                callback(cfg)

            except Exception as exc:
                log.error(f'(!) catched exception {type(exc)}')
                log.error(exc)


def add_conf_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        '-c', '--config', type=str,
        help='config file (conf/*.conf)',
    )

    parser.add_argument(
        '-s', '--spec', type=str,
        help='config specification file (conf/*.spec.conf)',
    )
