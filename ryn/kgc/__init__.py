# -*- coding: utf-8 -*-

from ryn.cli import main
from ryn.kgc import trainer
from ryn.common import helper
from ryn.common import logging

import click

import csv

from functools import partial
from tabulate import tabulate

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
    trainer.train_from_kwargs(**kwargs)


@click_kgc.command(name='evaluate')
@click.argument(
    'results', type=str, nargs=-1)
@click.option(
    '--split_dataset', type=str, required=True,
    help='path to the split dataset directory')
@click.option(
    '--out', type=str,
    help='directory to write the summary to')
def click_evaluate(out: str = None, **kwargs):
    """
    Evaluate a set of kgc models
    """
    results = trainer.evaluate_from_kwargs(**kwargs)

    # TODO move this to trainer.evaluate?

    sort_key = 3
    headers = ['name', 'model', 'hits@1', 'hits@10', 'A-MR']
    headers[sort_key] += ' *'

    def _save_rget(dic, *args, default=None):
        ls = list(args)[::-1]

        while ls:
            try:
                dic = dic[ls.pop()]
            except KeyError as exc:
                log.error(str(exc))
                return default

        return dic

    rows = []
    for eval_result, path in results:
        metrics = eval_result.metrics
        get = partial(_save_rget, metrics)

        rows.append([
            path.name,
            eval_result.model,
            get('hits_at_k', 'both', 'avg', '1', default=0) * 100,
            get('hits_at_k', 'both', 'avg', '10', default=0) * 100,
            get('adjusted_mean_rank', 'both', default=2),
        ])

    rows.sort(key=lambda r: r[sort_key], reverse=True)
    table = tabulate(rows, headers=headers)

    print()
    print(table)

    if out is not None:
        fname = 'evaluation'
        out = helper.path(
            out, create=True,
            message=f'writing {fname} to {{path_abbrv}}')

        with (out / (fname + '.txt')).open(mode='w') as fd:
            fd.write(table)

        with (out / (fname + '.csv')).open(mode='w') as fd:
            writer = csv.DictWriter(fd, fieldnames=headers)
            writer.writeheader()
            writer.writerows([dict(zip(headers, row)) for row in rows])
