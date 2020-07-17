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
    'split': {
        'create': split.create_from_args
    }
}


def args(parser):
    parser.add_argument(
        'cmd', type=str,
        help=f'one of {", ".join(CMDS)}', )

    parser.add_argument(
        'subcmd', type=str,
        help=f'graph:  ({", ".join(CMDS["graph"])}),', )

    parser.add_argument(
        '--seeds', type=int, nargs='+',
        help='random seeds', )

    parser.add_argument(
        '--ratios', type=int, nargs='+',
        help='ratio thresholds (cut at n-th relation for concepts)', )

    loader.add_graph_arguments(parser)


def main(args):
    log.info('running graphs')
    CMDS[args.cmd][args.subcmd](args)
