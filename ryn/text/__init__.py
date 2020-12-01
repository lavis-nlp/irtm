# -*- coding: utf-8 -*-
# fmt: off

import ryn
from ryn.cli import main
from ryn.text import data
from ryn.text import trainer
from ryn.text import evaluator
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
@click.option(
    '--dataset', type=str, required=True,
    help='path to ryn.text.data.Dataset')
@click.option(
    '--ratio', type=float, required=True,
    help='train/valid ratio for mapper (usually 0.7)')
@click.option(
    '--seed', type=int, required=True,
    help='seed for a deterministic split')
def cli(dataset: str = None, ratio: float = None, seed: int = None):
    """

    Open an interactive python shell

    dataset: path to text.data.Dataset directory

    """
    ds = data.Dataset.create(path=dataset, ratio=ratio, seed=seed)
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
@click.option(
    '--sqlite-database', type=str,
    help='path to sqlite text database')
@click.option(
    '--json-file', type=str,
    help='path to json file')
def transform(
        dataset: str = None,
        sqlite_database: str = None,
        json_file: str = None,
        **kwargs):
    """
    Transform a graph.split.Dataset to a text.data.Dataset
    """
    if json_file and sqlite_database:
        raise ryn.RynError('cannot provide both sqlite and json')

    if json_file:
        loader = 'json'
        loader_args = dict(fname=json_file)

    if sqlite_database:
        loader = 'sqlite'
        loader_args = dict(database=sqlite_database)

    dataset = split.Dataset.load(path=dataset)
    data.transform(
        dataset=dataset,
        loader=loader,
        loader_args=loader_args,
        **kwargs)


# shared options

_shared_options_mapper = [
    #
    # these are autogenerated by `ryn common ryaml click-arguments`
    #
    click.option('--valid-split', type=float),
    click.option('--wandb-args--project', type=str),
    click.option('--wandb-args--log-model', type=bool),
    click.option('--trainer-args--gpus', type=int),
    click.option('--trainer-args--max-epochs', type=int),
    click.option('--trainer-args--fast-dev-run', type=bool),
    click.option('--trainer-args--distributed-backend', type=str),
    click.option('--trainer-args--accumulate-grad-batches', type=int),
    click.option('--trainer-args--gradient-clip-val', type=int),
    click.option('--checkpoint-args--monitor', type=str),
    click.option('--checkpoint-args--save-top-k', type=int),
    click.option('--dataloader-train-args--num-workers', type=int),
    click.option('--dataloader-train-args--shuffle', type=bool),
    click.option('--dataloader-train-args--batch-size', type=int),
    click.option('--dataloader-valid-args--num-workers', type=int),
    click.option('--dataloader-valid-args--batch-size', type=int),
    click.option('--dataloader-test-args--num-workers', type=int),
    click.option('--dataloader-test-args--batch-size', type=int),
    click.option('--optimizer', type=str),
    click.option('--optimizer-args--lr', type=float),
    click.option('--text-encoder', type=str),
    click.option('--freeze-text-encoder', type=bool),
    click.option('--aggregator', type=str),
    click.option('--projector', type=str),
    click.option('--projector-args--input-dims', type=int),
    click.option('--projector-args--output-dims', type=int),
    click.option('--comparator', type=str),
    click.option('--split-dataset', type=str),
    click.option('--kgc-model', type=str),
    click.option('--text-dataset', type=str),
    click.option('--out', type=str),
]


# thanks https://stackoverflow.com/questions/40182157
def _add_options(options):
    def _proxy(fn):
        [option(fn) for option in reversed(options)]
        return fn
    return _proxy


@text.command()
@click.option(
    '--debug', is_flag=True,
    help='only test a model and do not log')
@click.option(
    '--offline', is_flag=True,
    help='run a complete training, but do so locally (wandb)')
@click.option(
    '-c', '--config', type=str, multiple=True,
    help='one or more configuration files')
@_add_options(_shared_options_mapper)
def train(*, config=None, **kwargs):
    """
    Train a mapper to align embeddings
    """
    trainer.train_from_kwargs(config=config, **kwargs)


@text.command()
@click.option(
    '--debug', is_flag=True,
    help='only test a model and do not log')
@click.option(
    '--offline', is_flag=True,
    help='run a complete training, but do so locally (wandb)')
@click.option(
    '--path', type=str, required=True,
    help='path to model directory')
@click.option(
    '--checkpoint', type=str, required=True,
    help='path to model checkpoint')
@_add_options(_shared_options_mapper)
def resume(**kwargs):
    """
    Resume training of a mapper
    """
    trainer.resume_from_kwargs(**kwargs)


@text.command()
@click.option(
    '--path', type=str, required=True,
    help='path to model directory')
@click.option(
    '--checkpoint', type=str, required=True,
    help='path to model checkpoint')
@click.option(
    '-c', '--config', type=str, multiple=True,
    help='one or more configuration files')
@click.option(
    '--debug', is_flag=True,
    help='run everything fast, do not write anything')
def evaluate(**kwargs):
    """
    Evaluate a mapper on the test split
    """
    evaluator.evaluate_from_kwargs(**kwargs)


@text.command()
@click.option(
    '-c', '--config', type=str, multiple=True,
    help='one or more configuration files')
@click.option(
    '--kgc-model', type=str, required=True,
    help='path to ryn.kgc.keen.Model')
@click.option(
    '--split-dataset', type=str, required=True,
    help='path to ryn.graphs.split.Dataset')
@click.option(
    '--out', type=str, required=True,
    help='where to write the results to')
@click.option(
    '--debug', is_flag=True,
    help='run everything fast, do not write anything')
def evaluate_baseline(**kwargs):
    """
    Evaluate a mapper where all projections are [1., ...]
    """
    evaluator.evaluate_baseline(**kwargs)


@text.command()
@click.option(
    '--root', type=str, required=True,
    help='')
@click.option(
    '-c', '--config', type=str, multiple=True,
    help='one or more configuration files')
@click.option(
    '--debug', is_flag=True,
    help='run everything fast, do not write anything')
def evaluate_all(**kwargs):
    """
    Run evaluations for all saved checkpoints
    """
    evaluator.evaluate_all(**kwargs)
