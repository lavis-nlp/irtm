# -*- coding: utf-8 -*-
# fmt: off

from irtm.cli import main
from irtm.kgc import trainer
from irtm.common import helper
from irtm.common import logging

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
    result = trainer.Result.load(result)

    print(f'{result.str_stats}')

    __import__("pdb").set_trace()

    # banner = '\n'.join((
    #     '',
    #     '-' * 20,
    #     ' IRTM KEEN CLIENT',
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
    help='path to irtm.graphs.split.Dataset folder')
@click.option(
    '--participate/--create', type=bool, default=False,
    help='for multi-process optimization')
def click_train(**kwargs):
    """
    Train a knowledge graph completion model
    """
    trainer.train_from_kwargs(**kwargs)


@click_kgc.command(name='resume')
@click.option(
    '--path', type=str, required=True,
    help='directory containing a config.json and optuna.db')
@click.option(
    '--split-dataset', type=str, required=True,
    help='path to irtm.graphs.split.Dataset folder')
def click_resume(**kwargs):
    """
    Resume hyperparameter search
    """
    trainer.resume_from_kwargs(**kwargs)


@click_kgc.command(name='evaluate')
@click.argument(
    'results', type=str, nargs=-1)
@click.option(
    '--split-dataset', type=str, required=True,
    help='path to the split dataset directory')
@click.option(
    '--out', type=str,
    help='directory to write the summary to')
@click.option(
    '--mode', type=str,
    help='one of validation, testing, all')
def click_evaluate(out: str = None, mode: str = None, **kwargs):
    """
    Evaluate a set of kgc models
    """
    results = trainer.evaluate_from_kwargs(mode=mode, **kwargs)
    trainer.print_results(results=results, out=out, mode=mode)
