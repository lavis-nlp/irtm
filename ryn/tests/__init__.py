# -*- coding: utf-8 -*-

import ryn
from ryn.common import logging

import inotify.adapters

import re
import unittest
import multiprocessing as mp
from datetime import datetime


log = logging.get('test')


def run(name: str = None):
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()
    tests = loader.discover(ryn.ENV.TEST_DIR, '*.py')

    if name:
        # e.g. 'graph.FindTests'
        tests = loader.loadTestsFromName(name)

    runner.run(tests)


# the unittest runner is not reloading modules
# so this is a work-around by forking
def _spawn():
    p = mp.Process(target=run, )
    p.start()
    p.join()


def main(args):
    if args.cmd == 'run':
        log.info(f'running tests: {args.name or "all"}')
        run(args.name)
        return

    if args.cmd == 'watch':
        print('\nover the mountain watching the watcher\n')
        ify = inotify.adapters.Inotify()

        ify.add_watch('.')
        ify.add_watch('../conf/')
        ify.add_watch('../ryn/')

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

        return

    raise ryn.RynError('unknown command: {args.cmd}')


# --- ryn interface


desc = 'run some tests'


def args(parser):
    parser.add_argument(
        'cmd', type=str, help='one of {run, watch}'
    )

    parser.add_argument(
        '--name', type=str, help='test case(s) to run'
    )
