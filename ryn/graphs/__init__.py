# -*- coding: utf-8 -*-

from ryn.cli import main
from ryn.common import logging
from ryn.graphs import split as split
from ryn.graphs import loader as loader

import click


log = logging.get('graphs')


# --- cli interface


@main.group(name='graphs')
def click_graphs():
    """
    Working with graphs
    """
    pass


@click_graphs.group(name='graph')
def click_graph():
    """
    Networkx graph abstractions
    """
    pass


@click_graph.command(name='cli')
@click.option(
    '--config', type=str,
    help='config file (conf/*.conf)')
@click.option(
    '--spec', type=str,
    help='config specification file (conf/*.spec.conf)')
@click.option(
    '--graphs', type=str, multiple=True,
    help='selection of graphs (names defined in --config)')
@click.option(
    '--uri', type=str, multiple=True,
    help=(
        'instead of -config -spec -graphs combination;'
        ' {provider}.{dataset} (e.g. oke.fb15k237-train)'))
def graph_cli(uri: str = None, single: bool = False, **kwargs):
    """

    Load graphs and drop into an interactive shell

    Provide either a --config --spec and --graphs
    or simply an --uri.

    """

    if uri:
        graphs = loader.load_graphs_from_uri(*uri)
    else:
        graphs = loader.load_graphs_from_conf(**kwargs)

    print()
    for name in graphs:
        print(f'loaded graph: {name}')

    banner = '\n'.join((
        '',
        '-' * 20,
        ' RYN GRAPH CLIENT',
        '-' * 20,
        '',
        'variables in scope:',
        '    graphs',
        '',
    ))

    import IPython
    IPython.embed(banner1=banner)


# ---


@click_graphs.group(name='split')
def click_split():
    """
    Create open world triple splits
    """
    pass


@click_split.command()
@click.option(
    '--seed', type=int, multiple=True, required=True,
    help='random seeds')
@click.option(
    '--ratio', type=int, multiple=True, required=True,
    help='ratio thresholds (cut at n-th relation for concepts)')
def create(**kwargs):
    """
    Create a graphs.split.Dataset from a graphs.graph.Graph

    It uses all combinations of seeds and ratios.

    """
    split.create_from_conf(**kwargs)


@click_split.command(name='cli')
@click.option(
    '--path', type=str, required=True,
    help='path to a graphs.split.Dataset directory')
def split_cli(path: str = None):
    """
    Load a split dataset
    """

    ds = split.Dataset.load(path=path)
    print(f'{ds}')

    banner = '\n'.join((
        '',
        '-' * 20,
        ' RYN DATASET CLIENT',
        '-' * 20,
        '',
        'variables in scope:',
        '    ds: Dataset',
        '',
    ))

    import IPython
    IPython.embed(banner1=banner)
