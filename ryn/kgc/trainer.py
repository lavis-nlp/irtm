# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import config
from ryn.kgc import pipeline
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging


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
        tracker=config.Tracker(
            cls='wandb',
            project='ryn-keen-proto',
            experiment=f'{keen_dataset.name}-{model}',
            reinit=True,
            offline=offline,
        ),
        model=config.Model(
            cls=model,
            embedding_dim=300,
            preferred_device='cuda',
            automatic_memory_optimization=True,
        ),
        optimizer=config.Optimizer(
            cls='Adagrad',
            lr=0.01,
        ),
        regularizer=config.Regularizer(
            cls='LpRegularizer',
            weight=0.1,
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
            batch_size=300,
        ),
        stopper=config.Stopper(
            cls='EarlyStopper',
            frequency=1,
            patience=10,
            relative_delta=0.0001,
        ),
        training_loop=config.TrainingLoop(
            cls='SLCWATrainingLoop',
        ),
        sampler=config.Sampler(
            cls='BasicNegativeSampler',
            num_negs_per_pos=5,
        ),
        training=config.Training(
            num_epochs=2,
            batch_size=128,
        ),
    )

    result = pipeline.single(
        config=conf,
        split_dataset=split_dataset,
        keen_dataset=keen_dataset)

    time = str(result.training_time.start).replace(' ', '_')
    result.save(ryn.ENV.KGC_DIR / split_dataset.name / f'{model}-{time}')


def train_from_cli(
        model: str = None,
        split_dataset: str = None,
        offline: bool = False):

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
