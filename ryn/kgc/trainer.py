# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc.config import Config
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import torch
import optuna
import numpy as np
from tqdm import tqdm as _tqdm
from tabulate import tabulate

import gc
import csv
import json
import pathlib
import textwrap
import dataclasses
from functools import partial
from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass

from typing import Any
from typing import List
from typing import Dict
from typing import Union
from typing import Iterable


log = logging.get('kgc.trainer')
tqdm = partial(_tqdm, ncols=80)


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
def _load_datasets(path: Union[str, pathlib.Path] = None):
    split_dataset = split.Dataset.load(path=path)
    keen_dataset = keen.Dataset.create(
        name=split_dataset.name,
        path=split_dataset.path,
        split_dataset=split_dataset)

    return split_dataset, keen_dataset


@helper.notnone
def _save_model(*, path: pathlib.Path = None, model=None):
    fname = 'model.ckpt'
    path = helper.path(
        path, exists=True,
        message=f'saving {fname} to {{path_abbrv}}')

    torch.save(model, str(path / fname))


# --------------------


@dataclass
class Time:

    @property
    def took(self) -> timedelta:
        return self.end - self.start

    start: datetime
    end: datetime

    @classmethod
    def create(K, dic: Dict[str, str]):
        return K(**{
            k: datetime.fromisoformat(v)
            for k, v in dic.items()})


@dataclass
class TrainingResult:

    created: datetime
    git_hash: str
    config: Config

    # metrics
    training_time: Time
    losses: List[float]

    model: torch.nn.Module

    # set by train()
    stopper: Any = None
    result_tracker: Any = None

    # set by Result.load
    wandb: Dict = None

    @property
    def str_stats(self):
        # TODO add metric_results

        s = f'training result for {self.config.model.cls}\n'
        s += textwrap.indent(
            f'created: {self.created}\n'
            f'git hash: {self.git_hash}\n'
            f'training took: {self.training_time.took}\n'
            f'dataset: {self.config.general.dataset}\n'
            f'seed: {self.config.general.seed}\n'
            '', ' ' * 2)

        return s

    @property
    def result_dict(self):
        dic = dict(
            created=self.created,
            git_hash=self.git_hash,
            # metrics
            training_time=dataclasses.asdict(self.training_time),
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
        fname = 'training.json'
        path = helper.path(
            path, exists=True,
            message=f'saving {fname} to {{path_abbrv}}')

        with (path / fname).open(mode='w') as fd:
            json.dump(self.result_dict, fd, default=str, indent=2)

    def save(self, path: Union[str, pathlib.Path]):
        path = helper.path(path, create=True)

        self.config.save(path)
        self._save_results(path)
        _save_model(path=path, model=self.model)

        with (path / 'summary.txt').open(mode='w') as fd:
            fd.write(self.str_stats)

    @classmethod
    def load(K, path: Union[str, pathlib.Path], load_model: bool = True):
        path = helper.path(
            path, exists=True,
            message='loading training results from {path_abbrv}')

        with (path / 'training.json').open(mode='r') as fd:
            raw = json.load(fd)

        model = None
        if load_model:
            model = torch.load(str(path / 'model.ckpt'))

        return K(**{**raw, **dict(
            training_time=Time.create(raw['training_time']),
            config=Config.load(path / 'config.json'),
            model=model,
        )})


# --------------------


@helper.notnone
def single(
        *,
        config: Config = None,
        split_dataset: split.Dataset = None,
        keen_dataset: keen.Dataset = None,
) -> TrainingResult:

    # preparation

    if not config.general.seed:
        # choice of range is arbitrary
        config.general.seed = np.random.randint(10**5, 10**7)

    helper.seed(config.general.seed)

    # initialization

    result_tracker = config.resolve(config.tracker)
    if not result_tracker.experiment:
        result_tracker = dataclasses.replace(
            result_tracker,
            experiment=f'{split_dataset.name}-{config.model.cls}')

    result_tracker.start_run()
    result_tracker.log_params(dataclasses.asdict(config))

    device = resolve_device(
        device_name=config.model.preferred_device)

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

    evaluator = config.resolve(
        config.evaluator,
    )

    optimizer = config.resolve(
        config.optimizer,
        params=model.get_grad_params(),
    )

    stopper = config.resolve(
        config.stopper,
        model=model,
        evaluator=evaluator,
        evaluation_triples_factory=keen_dataset.validation,
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
                clear_optimizer=True,
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

    # aggregation

    return TrainingResult(
        created=datetime.now(),
        git_hash=helper.git_hash(),
        config=config,
        # metrics
        training_time=training_time,
        losses=losses,
        # instances
        model=model,
        stopper=stopper,
        result_tracker=result_tracker,
    )


@helper.notnone
def _create_study(
        *,
        config: Config = None,
        out: pathlib.Path = None,
) -> optuna.Study:

    out.mkdir(parents=True, exist_ok=True)
    db_path = out / 'optuna.db'

    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M')
    study_name = f'{config.model.cls}-sweep-{timestamp}'

    log.info(f'create optuna study "{study_name}"')
    # TODO use direction="maximise"
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
        base: Config = None,
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


@helper.notnone
def train(
        *,
        config: Config = None,
        split_dataset: split.Dataset = None,
        keen_dataset: keen.Dataset = None,
        offline: bool = False,
) -> None:

    time = str(datetime.now()).replace(' ', '_')
    out = ryn.ENV.KGC_DIR / split_dataset.name / f'{config.model.cls}-{time}'
    config.save(out)

    multi(
        out=out,
        base=config,
        split_dataset=split_dataset,
        keen_dataset=keen_dataset)


@helper.notnone
def train_from_kwargs(
        *,
        config: str = None,
        split_dataset: str = None,
        offline: bool = False):
    log.info('running training from cli')

    if offline:
        log.warning('offline run!')

    split_dataset, keen_dataset = _load_datasets(path=split_dataset)

    print(f'\n{split_dataset}\n{keen_dataset}\n')

    config = Config.load(config)
    config.general.dataset = split_dataset.name

    train(
        config=config,
        split_dataset=split_dataset,
        keen_dataset=keen_dataset,
        offline=offline)


# --------------------


@dataclass
class EvaluationResult:

    model: str
    created: datetime
    git_hash: str

    # metrics
    evaluation_time: Time
    metrics: Dict

    @classmethod
    def load(K, path: Union[str, pathlib.Path]):
        path = helper.path(
            path, exists=True,
            message='loading {path_abbrv}')

        with path.open(mode='r') as fd:
            raw = json.load(fd)

        return K(
            model=raw['model'],
            created=raw['created'],
            git_hash=raw['git_hash'],
            evaluation_time=Time.create(raw['evaluation_time']),
            metrics=raw['metrics'],
        )


def _evaluate(train_result, keen_dataset):

    evaluator = train_result.config.resolve(
        train_result.config.evaluator,
    )

    ts = datetime.now()
    metrics = evaluator.evaluate(
        model=train_result.model,
        mapped_triples=keen_dataset.testing.mapped_triples,
        tqdm_kwargs=dict(
            position=1,
            ncols=80,
            leave=False,
        )
    )

    evaluation_time = Time(start=ts, end=datetime.now())

    return EvaluationResult(
        model=train_result.config.model.cls,
        created=datetime.now(),
        evaluation_time=evaluation_time,
        git_hash=helper.git_hash(),
        metrics=dataclasses.asdict(metrics),
    )


def _evaluate_wrapper(path, fname, train_result, keen_dataset):
    eval_result = _evaluate(train_result, keen_dataset)
    log.info(f'evaluation took: {eval_result.evaluation_time.took}')

    path = helper.path(path, message=f'writing {fname} to {{path_abbrv}}')
    with (path / fname).open(mode='w') as fd:
        json.dump(
            dataclasses.asdict(eval_result),
            fd, default=str, indent=2)

    return eval_result


@helper.notnone
def evaluate(
        *,
        glob: Iterable[pathlib.Path] = None,
        split_dataset: split.Dataset = None,
        keen_dataset: keen.Dataset = None):

    glob = list(glob)
    log.info(f'probing {len(glob)} directories')

    results = []
    for path in tqdm(glob):
        try:
            train_result = TrainingResult.load(path)
            assert train_result.config.general.dataset == split_dataset.name

        except (FileNotFoundError, NotADirectoryError) as exc:
            log.info(f'skipping {path.name}: {exc}')
            continue

        fname = 'evaluation.json'
        if (path / fname).is_file():
            eval_result = EvaluationResult.load(path / fname)

        else:
            eval_result = _evaluate_wrapper(
                path, fname, train_result, keen_dataset)

        results.append((eval_result, path))

    return results


@helper.notnone
def evaluate_from_kwargs(
        *,
        results: List[str] = None,
        split_dataset: str = None,
):

    split_dataset, keen_dataset = _load_datasets(path=split_dataset)

    eval_results = evaluate(
        glob=map(pathlib.Path, results),
        split_dataset=split_dataset,
        keen_dataset=keen_dataset)

    return eval_results


@helper.notnone
def print_results(*, results, out: Union[str, pathlib.Path] = None):

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
    sort_key = 4
    headers = ['name', 'model', 'val', 'hits@1', 'hits@10', 'A-MR']
    headers[sort_key] += ' *'

    for eval_result, path in results:
        train_result = TrainingResult.load(path, load_model=False)

        test_metrics = eval_result.metrics
        get_test = partial(_save_rget, test_metrics)

        rows.append([
            path.name,
            eval_result.model,
            train_result.stopper['results'][-1],
            get_test('hits_at_k', 'both', 'avg', '1', default=0) * 100,
            get_test('hits_at_k', 'both', 'avg', '10', default=0) * 100,
            get_test('adjusted_mean_rank', 'both', default=2),
        ])

    rows.sort(key=lambda r: r[sort_key], reverse=True)
    table = tabulate(rows, headers=headers)

    print()
    print(table)

    if out is not None:
        fname = 'evaluation'
        out = helper.path(
            out, create=True,
            message=f'writing {fname} txt/csv to {{path_abbrv}}')

        with (out / (fname + '.txt')).open(mode='w') as fd:
            fd.write(table)

        with (out / (fname + '.csv')).open(mode='w') as fd:
            writer = csv.DictWriter(fd, fieldnames=headers)
            writer.writeheader()
            writer.writerows([dict(zip(headers, row)) for row in rows])
