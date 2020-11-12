# -*- coding: utf-8 -*-

from ryn.cli import main
from ryn.text import data
from ryn.text import trainer
from ryn.graphs import split
from ryn.common import logging

import click


log = logging.get('text')


# --- cli interface


@main.group()
def text():
    """
    Process text data
    """
    log.info('running text')


@text.command()
@click.argument('dataset')
def cli(dataset: str = None):
    """

    Open an interactive python shell

    dataset: path to text.data.Dataset directory

    """
    ds = data.Dataset.load(path=dataset)
    print(f'\n{ds}')

    banner = '\n'.join((
        '',
        '-' * 20,
        ' RYN KEEN CLIENT',
        '-' * 20,
        '',
        'variables in scope:',
        '    ds: Dataset',
        '',
    ))

    import IPython
    IPython.embed(banner1=banner)


@text.command()
@click.option(
    '--dataset', type=str, required=True,
    help='path to ryn.graph.split.Dataset directory')
@click.option(
    '--database', type=str, required=True,
    help='path to sqlite text database')
@click.option(
    '--sentences', type=int, required=True,
    help='(max) number of sentences per entity')
@click.option(
    '--tokens', type=int, required=True,
    help='expected (max) number of tokens')
@click.option(
    '--model', type=str, required=True,
    help='huggingface pretrained model (e.g. bert-base-cased)')
@click.option(
    '--masked/--not-masked', default=False,
    help='whether to mask mentions')
@click.option(
    '--marked/--not-marked', default=False,
    help='whether to add marker tokens to mentions')
@click.option(
    '--suffix', type=str,
    help='suffix to append to the dataset directory')
def transform(
        dataset: str = None,
        **kwargs):
    """
    Transform a graph.split.Dataset to a text.data.Dataset
    """
    dataset = split.Dataset.load(path=dataset)
    data.transform(dataset=dataset, **kwargs)


@text.command()
@click.option(
    '--debug', is_flag=True,
    help='only test a model and do not log')
@click.option(
    '--split-dataset', type=str, required=True,
    help='path to ryn.graphs.split.Dataset directory')
@click.option(
    '--text-dataset', type=str, required=True,
    help='path to ryn.text.data.Dataset directory')
@click.option(
    '--kgc-model', type=str, required=True,
    help='path to ryn.kgc.keen.Model directory')
def train(**kwargs):
    """
    Train a mapper to align embeddings
    """
    trainer.train_from_cli(**kwargs)


@text.command()
def resume(**kwargs):
    """
    Resume training of a mapper
    """
    raise NotImplementedError()
    trainer.resume_from_cli(**kwargs)
