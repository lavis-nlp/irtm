# -*- coding: utf-8 -*-


from ryn.cli import main
from ryn.kgc import keen
from ryn.kgc import trainer
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
    '--path', type=str, required=True,
    help='path to a kgc.keen.Model directory')
def kgc_cli(path: str = None):

    m = keen.Model.load(path)
    print(f'\n{m}')

    banner = '\n'.join((
        '',
        '-' * 20,
        ' RYN KEEN CLIENT',
        '-' * 20,
        '',
        'variables in scope:',
        '    m: Model',
        '',
    ))

    import IPython
    IPython.embed(banner1=banner)


@click_kgc.command(name='train')
@click.option(
    '--model', type=str, required=True,
    help='model name (as registered with pykeen)')
@click.option(
    '--split-dataset', type=str, required=True,
    help='path to ryn.graphs.split.Dataset folder')
@click.option(
    '--offline', is_flag=True,
    help='do not sync wandb')
def click_train(**kwargs):
    trainer.train_from_cli(**kwargs)
