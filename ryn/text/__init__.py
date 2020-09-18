# -*- coding: utf-8 -*-

from ryn.text import data
from ryn.text import mapper
from ryn.common import logging


log = logging.get('text')


# --- ryn interface


desc = 'work with text data'


CMDS = {
    'transform': data._transform_from_args,
    'train': mapper.train_from_args,
}


def args(parser):
    parser.add_argument(
        'cmd', type=str,
        help=f'one of {", ".join(CMDS)}', )

    parser.add_argument(
        '--dataset', type=str,
        help='path to ryn.graph.split.Dataset directory')

    parser.add_argument(
        '--database', type=str,
        help='path to sqlite text database')

    parser.add_argument(
        '--sentences', type=int,
        help='(max) number of sentences per entity')

    parser.add_argument(
        '--tokens', type=int,
        help='expected (max) number of tokens')

    parser.add_argument(
        '--model', type=str,
        help='huggingface pretrained model')


def main(args):
    log.info('running text')
    CMDS[args.cmd](args)
