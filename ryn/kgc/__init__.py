# -*- coding: utf-8 -*-


from ryn.cli import main
from ryn.kgc import trainer
from ryn.kgc import pipeline
from ryn.common import helper
from ryn.common import logging

import click


log = logging.get('kgc')


@main.group(name='kgc')
def click_kgc():
    """
    Knowledge graph completion models
    """
    pass


@click_kgc.command(name='cli')
@click.option(
    '--result', type=str, required=True,
    help='path to a trained kgc model directory')
def kgc_cli(result: str = None):
    """
    Open a shell and load a result
    """

    result = helper.path(result, exists=True)
    result = pipeline.Result.load(result)

    print(f'{result.str_stats}')

    __import__("pdb").set_trace()

    # banner = '\n'.join((
    #     '',
    #     '-' * 20,
    #     ' RYN KEEN CLIENT',
    #     '-' * 20,
    #     '',
    #     'variables in scope:',
    #     '    m: Model',
    #     '',
    # ))

    # import IPython
    # IPython.embed(banner1=banner)


@click_kgc.command(name='train')
@click.option(
    '--config', type=str, required=True,
    help='json (see conf/kgc/*json)')
@click.option(
    '--split-dataset', type=str, required=True,
    help='path to ryn.graphs.split.Dataset folder')
@click.option(
    '--offline', is_flag=True,
    help='do not sync wandb')
def click_train(**kwargs):
    """
    Train a knowledge graph completion model
    """
    trainer.train_from_cli(**kwargs)
