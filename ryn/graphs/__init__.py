# -*- coding: utf-8 -*-

from ryn.graphs import graph
from ryn.graphs import loader
from ryn.common import logging


log = logging.get('graphs')


# --- ryn interface


desc = 'analyse paths through the networks'


CMDS = {
    'graph': {
        'cli': graph._cli,
    }
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

    parser.add_argument(
        '--path', type=str, default=None,
        help='path to a file or directory'
    )

    parser.add_argument(
        '--path-length', type=int, default=None,
        help='desired path length'
    )

    parser.add_argument(
        '--comp-procs', type=int, default=16,
        help='computation worker processes (cpu heavy)'
    )

    parser.add_argument(
        '--cons-procs', type=int, default=4,
        help='consolidation worker processes (ram heavy)'
    )


def main(args):
    log.info('running graphs')
    CMDS[args.cmd][args.subcmd](args)
