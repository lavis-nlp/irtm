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
    '--suffix', type=str,
    help='suffix to append to the dataset directory')
def transform(
        dataset: str = None,
        database: str = None,
        sentences: int = None,
        tokens: int = None,
        model: str = None,
        suffix: str = None, ):
    """
    Transform a graph.split.Dataset to a text.data.Dataset
    """
    dataset = split.Dataset.load(path=dataset)
    data.transform(
        dataset=dataset,
        database=database,
        sentences=sentences,
        tokens=tokens,
        model=model,
        suffix=suffix, )


@text.command()
@click.option(
    '--debug', is_flag=True,
    help='only test a model and do not log')
def train(debug: bool = None):
    """
    Train a mapper to align embeddings
    """
    trainer.train_from_cli(debug=debug)
