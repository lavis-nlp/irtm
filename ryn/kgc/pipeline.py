# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import config
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import torch
import optuna
import numpy as np

import json
import pathlib
import textwrap
import dataclasses
from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass

from typing import Any
from typing import List
from typing import Union


log = logging.get('kgc.pipeline')


@helper.notnone
def resolve_device(*, device_name: str = None):
    if device_name not in ('cuda', 'cpu'):
        raise ryn.RynError(f'unknown device option: "{device_name}"')

    if not torch.cuda.is_available() and device_name == 'cuda':
        log.error('cuda is not available; falling back to cpu')
        device_name = 'cpu'

    device = torch.device(device_name)
    log.info(f'resolved device, running on {device}')

    return device


@dataclass
class Time:

    @property
    def took(self) -> timedelta:
        return self.end - self.start

    start: datetime
    end: datetime


@dataclass
class Result:

    created: datetime
    git_hash: str
    config: config.Config

    # metrics
    training_time: Time
    evaluation_time: Time
    losses: List[float]

    # instances (may be None when recreating from disk)
    # being lazy: not annotating with all the pykeen classes
    model: Any
    stopper: Any = None
    result_tracker: Any = None

    @property
    def str_stats(self):
        # TODO add metric_results

        s = f'training result for {self.model.name}\n'
        s += textwrap.indent(
            f'created: {self.created}\n'
            f'git hash: {self.git_hash}\n',
            f'training took: {self.training_time.took}\n'
            f'evaluation took: {self.evaluation_time.took}\n'
            f'dataset: {self.config.general.dataset}\n'
            f'seed: {self.config.general.seed}\n'
            ' ' * 2)

        return s

    @property
    def result_dict(self):
        dic = dict(
            created=self.created,
            git_hash=self.git_hash,
            # metrics
            training_time=dataclasses.asdict(self.training_time),
            evaluation_time=dataclasses.asdict(self.evaluation_time),
            losses=self.losses,
            stopper=self.stopper.get_summary_dict(),
        )

        # tracking
        wandb_run = self.result_tracker.run
        dic['wandb'] = dict(
            id=wandb_run.id,
            dir=wandb_run.dir,
            path=wandb_run.path,
            name=wandb_run.name,
            project=wandb_run.project,
            offline=wandb_run.offline,
            sweep_id=wandb_run.sweep_id,
        )

        if not wandb_run.offline:
            dic['wandb'].update(dict(
                project_url=wandb_run.get_project_url(),
                sweep_url=wandb_run.get_sweep_url(),
                url=wandb_run.get_url(),
            ))

        return dic

    def _save_results(self, path):
        _path_abbrv = f'{path.parent.name}/{path.name}'
        fname = 'result.json'
        log.info(f'saving results to {_path_abbrv}/{fname}')
        with (path / fname).open(mode='w') as fd:
            json.dump(self.result_dict, fd, default=str, indent=2)

    def _save_model(self, path):
        _path_abbrv = f'{path.parent.name}/{path.name}'
        fname = 'model.ckpt'
        log.info(f'saving results to {_path_abbrv}/{fname}')
        torch.save(self.model, str(path / fname))

    def save(self, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save(path)
        self._save_results(path)
        self._save_model(path)

        with (path / 'summary.txt').open(mode='w') as fd:
            fd.write(self.str_stats)

    @staticmethod
    def load(self):
        raise NotImplementedError


@helper.notnone
def single(
        *,
        config: config.Config = None,
        split_dataset: split.Dataset = None,
        keen_dataset: keen.Dataset = None,
) -> Result:

    # preparation

    if not config.general.seed:
        # choice of range is arbitrary
        config.general.seed = np.random.randint(10**5, 10**7)

    helper.seed(config.general.seed)

    # initialization

    device = resolve_device(
        device_name=config.model.preferred_device)

    result_tracker = config.resolve(config.tracker)
    result_tracker.start_run()
    result_tracker.log_params(dataclasses.asdict(config))

    # target filtering for ranking losses is enabled by default
    loss = config.resolve(
        config.loss,
    )

    regularizer = config.resolve(
        config.regularizer,
        device=device,
    )

    model = config.resolve(
        config.model,
        loss=loss,
        regularizer=regularizer,
        random_seed=config.general.seed,
        triples_factory=keen_dataset.training,
        preferred_device=device,
    )

    optimizer = config.resolve(
        config.optimizer,
        params=model.get_grad_params(),
    )

    evaluator = config.resolve(
        config.evaluator,
    )

    log.error('setting evaluation batch size manually!')
    stopper = config.resolve(
        config.stopper,
        model=model,
        evaluator=evaluator,
        evaluation_triples_factory=keen_dataset.training,
        result_tracker=result_tracker,
        evaluation_batch_size=200,
    )

    training_loop = config.resolve(
        config.training_loop,
        model=model,
        optimizer=optimizer,
        negative_sampler_cls=config.sampler.constructor,
        negative_sampler_kwargs=config.sampler.kwargs,
    )

    # kindling

    ts = datetime.now()

    # losses = training_loop(...
    losses = training_loop.train(**{
        **dataclasses.asdict(config.training),
        **dict(
            stopper=stopper,
            result_tracker=result_tracker,
            clear_optimizer=False,
        )
    })

    training_time = Time(start=ts, end=datetime.now())
    ts = datetime.now()

    # TODO evaluator -> metric_results

    evaluation_time = Time(start=ts, end=datetime.now())

    return Result(
        created=datetime.now(),
        git_hash=helper.git_hash(),
        config=config,
        # metrics
        training_time=training_time,
        evaluation_time=evaluation_time,
        losses=losses,
        # instances
        model=model,
        stopper=stopper,
        result_tracker=result_tracker,
    )


@helper.notnone
def multi(
        *,
        base: config.Config = None,
        out: pathlib.Path = None,
        **kwargs
) -> None:

    # Optuna lingo:
    #   Trial: A single call of the objective function
    #   Study: An optimization session, which is a set of trials
    #   Parameter: A variable whose value is to be optimized
    # Parameter spaces:
    #      categorical -> List[Any]
    #              int -> Tuple[lower: int, upper: int]
    #          uniform -> Tuple[lower: float, upper: float]
    #       loguniform -> Tuple[lower: float, upper: float]
    # discrete_uniform -> Tuple[lower: float, upper: float, step: float]

    # TODO optuna config
    # TODO study db place
    study = optuna.create_study(storage='sqlite:///data/study.db')

    def _objective(trial):

        # obtain optuna suggestions
        config = base.from_trial(trial)
        name = f'{config.tracker.experiment}-{trial.number}'

        # update configuration
        tracker = dataclasses.replace(config.tracker, experiment=name)
        config = dataclasses.replace(config, tracker=tracker)

        # run training
        result = single(config=config, **kwargs)

        best_metric = result.stopper.best_metric
        log.info(f'! trial {trial.number} finished: '
                 f'best metric = {best_metric}')

        # min optimization
        result.save(out / f'trial-{trial.number:04d}')
        return -best_metric

    study.optimize(_objective, n_trials=100)
    log.info('finished study')
