# -*- coding: utf-8 -*-

import ryn
from ryn.cli import main
from ryn.common import logging

import click
import inotify.adapters

import re
import unittest
import multiprocessing as mp
from datetime import datetime


log = logging.get('test')


def _run(name: str = None):
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()
    tests = loader.discover(ryn.ENV.TEST_DIR, '*.py')

    if name:
        # e.g. 'graph.FindTests'
        tests = loader.loadTestsFromName(name)

    runner.run(tests)


# --- cli interface


@main.group(name='tests')
def click_tests():
    """
    Run unit tests
    """
    pass


@click_tests.command()
@click.option(
    '--name', type=str,
    help='test case(s) to run')
def run(name: str = None):
    log.info(f'running tests: {name or "all"}')
    _run(name=name)


@click_tests.command()
def watch():
    raise ryn.RynError('disabled for the moment')

    # the unittest runner is not reloading modules
    # so this is a work-around by forking
    def _spawn():
        p = mp.Process(target=_run, )
        p.start()
        p.join()

    print('\nover the mountain watching the watcher\n')
    ify = inotify.adapters.Inotify()

    ify.add_watch(str(ryn.ENV.SRC_DIR))
    ify.add_watch(str(ryn.ENV.CONF_DIR))

    for evt in ify.event_gen(yield_nones=False):
        _, types, path, fname = evt

        if any((
                not re.match(r'^[a-zA-Z]', fname),
                'IN_CLOSE_WRITE' not in types)):
            continue

        ts = datetime.now()
        print(f'[{ts}]', types, path, fname)

        _spawn()
        ts = datetime.now()
        print(f'\n[{ts}] waiting for file changes')
