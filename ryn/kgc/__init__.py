# -*- coding: utf-8 -*-


from ryn.cli import main
from ryn.kgc import keen
from ryn.common import logging

import click


log = logging.get('kgc')


@main.group(name='kgc')
def click_kgc():
    """
    Knowledge graph completion models
    """
    log.info('kgc called')


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
def click_train():
    log.info('train called')
    keen.run()
