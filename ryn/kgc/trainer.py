# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import config
from ryn.kgc import pipeline
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

from datetime import datetime


log = logging.get('kgc.trainer')


@helper.notnone
def train(
        *,
        model: str = None,
        split_dataset: split.Dataset = None,
        keen_dataset: keen.Dataset = None,
        offline: bool = False,
) -> None:

    conf = config.Config(
        general=config.General(
            dataset=split_dataset.name,
        ),
        optuna=config.Optuna(
            study_name=f'{model}-keen-sweep',
            trials=100,
            maximise=True,  # hits@10
        ),
        tracker=config.Tracker(
            cls='wandb',
            project='ryn-keen',
            experiment=f'{keen_dataset.name}-{model}',
            reinit=True,
            offline=offline,
        ),
        model=config.Model(
            cls=model,
            embedding_dim=config.IntSuggestion(
                low=50, high=500, step=50, initial=250),
            automatic_memory_optimization=True,
            preferred_device='cuda',
        ),
        optimizer=config.Optimizer(
            cls='Adagrad',
            lr=config.FloatSuggestion(
                low=1e-3, high=1e-1, log=True, initial=1e-2),
        ),
        regularizer=config.Regularizer(
            cls='LpRegularizer',
            weight=config.FloatSuggestion(low=1e-2, high=1e-1, log=True),
            p=2.0,
            normalize=True,
        ),
        loss=config.Loss(
            cls='MarginRankingLoss',
            margin=1.0,
            reduction='mean',
        ),
        evaluator=config.Evaluator(
            cls='RankBasedEvaluator',
        ),
        stopper=config.Stopper(
            cls='EarlyStopper',
            frequency=20,
            patience=20,
            relative_delta=1e-2,
        ),
        training_loop=config.TrainingLoop(
            cls='SLCWATrainingLoop',
        ),
        sampler=config.Sampler(
            cls='BasicNegativeSampler',
            num_negs_per_pos=config.IntSuggestion(
                low=1, high=20, step=1, initial=5),
        ),
        training=config.Training(
            num_epochs=2000,
            batch_size=250,
        ),
    )

    time = str(datetime.now()).replace(' ', '_')
    out = ryn.ENV.KGC_DIR / split_dataset.name / f'{model}-{time}'

    pipeline.multi(
        base=conf,
        out=out,
        split_dataset=split_dataset,
        keen_dataset=keen_dataset)


def train_from_cli(
        model: str = None,
        split_dataset: str = None,
        offline: bool = False):
    log.info('running training from cli')

    if offline:
        log.warning('offline run!')

    split_dataset = split.Dataset.load(path=split_dataset)
    keen_dataset = keen.Dataset.create(
        name=split_dataset.name,
        path=split_dataset.path,
        split_dataset=split_dataset)

    print(f'\n{split_dataset}\n{keen_dataset}\n')

    train(
        model=model,
        split_dataset=split_dataset,
        keen_dataset=keen_dataset,
        offline=offline)
