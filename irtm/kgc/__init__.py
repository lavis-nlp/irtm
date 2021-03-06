# -*- coding: utf-8 -*-
# fmt: off

from irtm.cli import main
from irtm.kgc import trainer
from irtm.kgc import evaluator
from irtm.common import helper

import click

import logging


log = logging.getLogger(__name__)


@main.group(name='kgc')
def click_kgc():
    """
    Closed-world knowledge graph completion
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
    help='yaml (see conf/kgc/*yml)')
@click.option(
    '--dataset', type=str, required=True,
    help='path to irt.Dataset folder')
@click.option(
    '--out', type=str, required=True,
    help='where to write the model, configuration, etc. to')
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
    help='directory containing a config.yml and optuna.db')
@click.option(
    '--dataset', type=str, required=True,
    help='path to irt.Dataset folder')
def click_resume(**kwargs):
    """
    Resume hyperparameter search
    """
    trainer.resume_from_kwargs(**kwargs)


@click_kgc.command(name='evaluate')
@click.argument(
    'results', type=str, nargs=-1)
@click.option(
    '--dataset', type=str, required=True,
    help='path to the irt.Dataset directory')
@click.option(
    '--out', type=str,
    help='directory to write the summary to')
@click.option(
    '--mode', type=str, default='validation',
    help='either validation or testing')
def click_evaluate(out: str = None, mode: str = None, **kwargs):
    """
    Evaluate a set of kgc models
    """
    results = evaluator.evaluate_from_kwargs(mode=mode, **kwargs)
    evaluator.print_results(results=results, out=out, mode=mode)
