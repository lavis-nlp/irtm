# -*- coding: utf-8 -*-


from ryn.embers import oke
from ryn.common import config
from ryn.common import logging


log = logging.get('embers')


# --- ryn interface


desc = 'work with graph embeddings'


CMDS = {
    'oke': {
        'train': oke.train_from_args,
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
            f'oke:  ({", ".join(CMDS["oke"])}),'
        ))


def main(args):
    log.info('running embers')
    CMDS[args.cmd][args.subcmd](args)
