#!/usr/bin/env python


from ryn import common
from ryn import logging

import inotify.adapters

import re
import unittest
import multiprocessing as mp
from datetime import datetime


log = logging.get('test')


t_dir = 'tests/'
desc = 'run the test suite'


def run(name: str = None):
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()
    tests = loader.discover(t_dir, '*.py')

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


def args(parser):
    parser.add_argument(
        'cmd', type=str, help='one of {run, watch}'
    )

    parser.add_argument(
        '--name', type=str, help='test case(s) to run'
    )


def main(args):
    if args.cmd == 'run':
        log.info(f'running tests: {args.name or "all"}')
        run(args.name)
        return

    if args.cmd == 'watch':
        print('\nover the mountain watching the watcher\n')
        ify = inotify.adapters.Inotify()

        ify.add_watch(t_dir)
        ify.add_watch('conf/')
        ify.add_watch('ryn/')

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

    raise a_common.RynError('unknown command: {args.cmd}')
