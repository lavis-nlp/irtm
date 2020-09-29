# -*- coding: utf-8 -*-

from ryn.common import config
from ryn.common import logging

from ryn.kgc import keen


log = logging.get('kgc')


# --- ryn interface


desc = 'work with knowledge graph embeddings'


CMDS = {
    'keen': {
        'train': keen.train_from_args,
        'cli': keen._cli,
    }
}


def args(parser):
    config.add_conf_arguments(parser)

    parser.add_argument(
        'cmd', type=str,
        help=f'one of {", ".join(CMDS)}')

    parser.add_argument(
        'subcmd', type=str,
        help=(
            f'keen:  ({", ".join(CMDS["keen"])}),'
        ))

    parser.add_argument(
        '--path', type=str,
        help='path to file or directory')


def main(args):
    log.info('running kgc')
    CMDS[args.cmd][args.subcmd](args)
