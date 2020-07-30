# -*- coding: utf-8 -*-


from ryn.common import logging
from streamlit import cli
import sys


log = logging.get('app')


class Context:

    def run(self):
        raise NotImplementedError()


# --- ryn interface


desc = 'handle streamlit instances'


def args(parser):
    pass


# streamlit internally relies on click contexts
# and there is a bug if there is no parent context.
# So: I just steal the streamlit click context


@cli.main.command()
def ryn():
    from ryn.app import app
    cli._main_run(app.__file__)


def main(args):
    log.info('running streamlit')

    # clear argv to obtain a clean click state
    sys.argv = [sys.argv[0], 'ryn']
    cli.main()
