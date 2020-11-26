# -*- coding: utf-8 -*-
# fmt: off

from ryn.cli import main
from ryn.common import logging

import sys
from dataclasses import dataclass

import streamlit
from streamlit import cli as streamlit_cli
from streamlit import logger as streamlit_logger

log = logging.get('app')


# streamlit registers a streamhandler per logger...
# https://github.com/streamlit/streamlit/blob/a8a90461bd77359ad812d847943b0e421ac85de9/lib/streamlit/logger.py#L66
streamlit.logger.set_log_level('INFO')
for logger in streamlit_logger.LOGGERS.values():
    logger.removeHandler(logger.streamlit_console_handler)


# ---


@dataclass
class Widgets:

    @staticmethod
    def read(widget, name, vals, mapper):
        VAL_ALL = 'all'

        vals = [VAL_ALL] + vals
        val = widget(name, vals)

        return None if val == VAL_ALL else mapper(val)


class Context:

    def run(self):
        raise NotImplementedError()


# --- cli interface


# streamlit internally relies on click contexts
@streamlit_cli.main.command()
def ryn():
    from ryn.app import app
    streamlit_cli._main_run(app.__file__)


@main.command()
def streamlit():
    """
    Run a streamlit app instance
    """
    log.info('running streamlit')

    # clear argv to obtain a clean click state
    sys.argv = [sys.argv[0], 'ryn']
    streamlit_cli.main()
