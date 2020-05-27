# -*- coding: utf-8 -*-

from ryn.graphs import graph
from ryn.graphs import split
from ryn.graphs import loader
from ryn.common import logging


log = logging.get('graphs')


# --- ryn interface


desc = 'analyse paths through the networks'


CMDS = {
    'graph': {
        'cli': graph._cli,
    },
}


def args(parser):
    parser.add_argument(
        'cmd', type=str,
        help=f'one of {", ".join(CMDS)}')

    parser.add_argument(
        'subcmd', type=str,
        help=(
            f'graph:  ({", ".join(CMDS["graph"])}),'
        ))

    loader.add_graph_arguments(parser)


def main(args):
    log.info('running graphs')
    CMDS[args.cmd][args.subcmd](args)
