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

import gc
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


@helper.notnone
def _save_model(*, path: pathlib.Path = None, model=None):
    _path_abbrv = f'{path.parent.name}/{path.name}'
    fname = 'model.ckpt'
    log.info(f'saving results to {_path_abbrv}/{fname}')
    torch.save(model, str(path / fname))


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

        s = 'training result for {self.config.model.cls}\n'
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
            offline=True,
        )

        if not hasattr(wandb_run, 'offline'):
            dic['wandb'].update(dict(
                url=wandb_run.url,
                offline=False,
            ))

        return dic

    def _save_results(self, path):
        _path_abbrv = f'{path.parent.name}/{path.name}'
        fname = 'result.json'
        log.info(f'saving results to {_path_abbrv}/{fname}')
        with (path / fname).open(mode='w') as fd:
            json.dump(self.result_dict, fd, default=str, indent=2)

    def save(self, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save(path)
        self._save_results(path)
        _save_model(path=path, model=self.model)

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

    stopper = config.resolve(
        config.stopper,
        model=model,
        evaluator=evaluator,
        evaluation_triples_factory=keen_dataset.training,
        result_tracker=result_tracker,
    )

    training_loop = config.resolve(
        config.training_loop,
        model=model,
        optimizer=optimizer,
        negative_sampler_cls=config.sampler.constructor,
        negative_sampler_kwargs=config.sampler.kwargs,
    )

    # training

    ts = datetime.now()

    try:

        losses = training_loop.train(**{
            **dataclasses.asdict(config.training),
            **dict(
                stopper=stopper,
                result_tracker=result_tracker,
                clear_optimizer=False,
            )
        })

    except RuntimeError as exc:
        log.error(f'training error: "{exc}"')
        log.error('sweeping training loop memory up under the rug')

        gc.collect()
        training_loop.optimizer.zero_grad()
        training_loop._free_graph_and_cache()

        raise exc

    training_time = Time(start=ts, end=datetime.now())
    result_tracker.log_metrics(
        prefix='validation',
        metrics=dict(best=stopper.best_metric, metric=stopper.metric),
        step=stopper.best_epoch)

    # evaluation

    ts = datetime.now()
    # TODO evaluator -> metric_results
    evaluation_time = Time(start=ts, end=datetime.now())

    # aggregation

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
def _create_study(
        *,
        config: config.Config = None,
        out: pathlib.Path = None,
) -> optuna.Study:

    out.mkdir(parents=True, exist_ok=True)
    db_path = out / 'optuna.db'

    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M')
    study_name = f'{config.optuna.study_name}-{timestamp}'

    log.info(f'create optuna study "{study_name}"')
    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{db_path}',
    )

    # if there are any initial values to be set,
    # create and enqueue a custom trial

    params = {
        k: v.initial for k, v in config.suggestions.items()
        if v.initial is not None}

    if params:
        log.info('setting initial study params: ' + ', '.join(
            f'{k}={v}' for k, v in params.items()))
        study.enqueue_trial(params)

    return study


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
    assert base.optuna, 'no optuna config found'

    def objective(trial):

        # obtain optuna suggestions
        config = base.suggest(trial)
        name = f'{config.tracker.experiment}-{trial.number}'
        path = out / f'trial-{trial.number:04d}'

        # update configuration
        tracker = dataclasses.replace(config.tracker, experiment=name)
        config = dataclasses.replace(config, tracker=tracker)

        # run training
        try:
            result = single(config=config, **kwargs)
        except RuntimeError as exc:
            msg = f'objective: got runtime error "{exc}"'
            log.error(msg)

            # post mortem (TODO last model checkpoint)
            config.save(path)
            raise ryn.RynError(msg)

        best_metric = result.stopper.best_metric
        log.info(f'! trial {trial.number} finished: '
                 f'best metric = {best_metric}')

        # min optimization
        result.save(path)
        return -best_metric if base.optuna.maximise else best_metric

    study = _create_study(config=base, out=out)

    study.optimize(
        objective,
        n_trials=base.optuna.trials,
        gc_after_trial=True,
        catch=(ryn.RynError, ),
    )

    log.info('finished study')
