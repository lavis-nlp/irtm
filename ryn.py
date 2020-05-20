#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import test

from ryn import graphs
from ryn import embers

from ryn import RynError
from ryn.common import logging

import sys
import argparse


log = logging.get('ryn')


mods = {
    'test': test,
    'graphs': graphs,
    'embers': embers,
}


def term(message: str = None, status=0):
    if message:
        if status == 0:
            log.info(message)
        else:
            log.error(message)

    log.info('exiting')
    log.info('+' * 80)
    sys.exit(status)


def print_help():
    print('RYN - working with knowledge graphs\n')
    print('usage: ./ryn.py CMD [ARGS]\n')
    print('  possible values for CMD:')
    print('       help: print this message')
    for name, mod in mods.items():
        print(f'    {name:>7s}: {mod.desc}')

    print('\nto get CMD specific help type ./ryn.py CMD --help')
    print('e.g. ./ryn.py ember --help')


def main():

    # --- select module

    try:
        cmd = sys.argv[1]
        if cmd in {'help', '-h', '--help'}:
            print_help()
            return

        mod = mods[cmd]
        log.info(f'attempting to run "{cmd}"')

    except IndexError:
        print('please provide a command')
        print_help()
        term(message='terminating upon argument error', status=2)

    except KeyError:
        print(f'unknown command: "{cmd}"', file=sys.stderr)
        print_help()
        term(message='terminating upon argument error', status=2)

    # --- load modules

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('_')  # cmd placeholder (sys.argv[1])
        mod.args(parser)

        args = parser.parse_args()
        mod.main(args)

    except RynError as err:
        print('An error occured:')
        print(str(err))
        term(message=f'internal error: {err}', status=2)

    except KeyboardInterrupt:
        print('received keyboard interrupt')
        term(message='terminating upon keyboard interrupt', status=0)


if __name__ == '__main__':
    main()
    term(message='terminating gracefully', status=0)
