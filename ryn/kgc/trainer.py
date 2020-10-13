# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import pipeline
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import copy
import json
from datetime import datetime
from dataclasses import dataclass

log = logging.get('kgc.trainer')


RANGE_TYPES = dict(
    model_kwargs_ranges=dict(embedding_dim=int),
    negative_sampler_kwargs_ranges=dict(num_negs_per_pos=int),
    training_kwargs_ranges=dict(batch_size=int),
)


@dataclass
class Config:

    @property
    def name(self) -> str:
        return f'{self.model}-{self.dataset_name}'

    model: str

    graph_name: str
    dataset_name: str


@helper.notnone
def train(
        *,
        config: Config = None,
        split_dataset: split.Dataset = None,
        keen_dataset: keen.Dataset = None,
        **kwargs):  # pykeen hpo_pipeline arguments

    metadata = {
        'pipeline': copy.deepcopy(kwargs),
        'config': config.__dict__
    }

    # You need to provide the data type to the
    # model_kwargs_ranges. They are not json
    # serializable. As such - they need to be set afterwards.
    for key, options in RANGE_TYPES.items():
        if key not in kwargs:
            continue

        for optkey, optval in options.items():
            if optkey not in kwargs[key]:
                continue

            kwargs[key][optkey]['type'] = int

    # execute hpo pipeline
    res = pipeline.hpo_pipeline(
        dataset=keen_dataset,
        **kwargs)

    # persist

    fname = '-'.join((
        config.model,
        str(datetime.now().strftime(keen.DATEFMT)), ))

    path = ryn.ENV.KGC_DIR / split_dataset.path.name / fname
    log.info(f'writing results to {path}')

    with (path / 'metadata.json').open(mode='w') as fd:
        json.dump(metadata, fd)

    res.save_to_directory(str(path))

    # TODO log seed etc. (set by pykeen) (look what res entails)


def train_from_cli(
        model: str = None,
        debug: bool = False):

    if debug:
        # TODO patch pykeen WANDBResultTracker to allow offline runs
        raise ryn.RynError('currently disabled')
        log.warning('phony debug run!')

    path = ryn.ENV.SPLIT_DIR / 'oke.fb15k237_30061990_50/'

    split_dataset = split.Dataset.load(path=path)
    keen_dataset = keen.Dataset.create(
        path=split_dataset.path,
        split_dataset=split_dataset)

    config = Config(
        model=model,
        dataset_name=split_dataset.path.name,
        graph_name=split_dataset.g.name)

    print(f'\nrunning {config.name} {split_dataset.path.name}\n')
    print(f'{keen_dataset}')

    kwargs = dict(

        # GENERAL

        model=config.model,

        result_tracker='wandb',
        result_tracker_kwargs=dict(
            project='ryn-keen',
            experiment=config.name,
            reinit=True),

        stopper='early',
        stopper_kwargs=dict(
            frequency=50,
            patience=10,
            relative_delta=0.002),

        # HPO

        n_trials=30,

        model_kwargs_ranges=dict(
            embedding_dim=dict(low=100, high=500, q=50)),

        training_kwargs_ranges=dict(
            batch_size=dict(low=64, high=1024, q=64)),

        negative_sampler_kwargs_ranges=dict(
            num_negs_per_pos=dict(low=2, high=20, q=2)),

    )

    train(
        split_dataset=split_dataset,
        keen_dataset=keen_dataset,
        config=config,
        **kwargs)
