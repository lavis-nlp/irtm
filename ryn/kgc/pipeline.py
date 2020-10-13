# -*- coding: utf-8 -*-

from ryn.kgc import keen
from ryn.common import logging

import torch
from pykeen import hpo
from pykeen.hpo import pruners as hpo_pruners
from pykeen.hpo import samplers as hpo_samplers

from pykeen import utils as keen_utils
from pykeen import losses as keen_losses
from pykeen import models as keen_models
from pykeen import version as keen_version
from pykeen import evaluation as keen_eval
from pykeen import triples as keen_triples
from pykeen import stoppers as keen_stoppers
from pykeen import sampling as keen_sampling
from pykeen import training as keen_training
from pykeen import trackers as keen_trackers
from pykeen import optimizers as keen_optims
from pykeen import regularizers as keen_regs
from pykeen import pipeline as keen_pipeline
from pykeen.models import base as keen_models_base

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage

import os
from dataclasses import dataclass

from typing import Any
from typing import Type
from typing import Dict
from typing import Union
from typing import Mapping
from typing import Optional


log = logging.get('kgc.pipeline')


def _get_kwargs(
    trial: optuna.Trial,
    prefix: str,
    *,
    default_kwargs_ranges: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    kwargs_ranges: Optional[Mapping[str, Any]] = None,
):
    _kwargs_ranges = dict(default_kwargs_ranges)
    if kwargs_ranges is not None:
        _kwargs_ranges.update(kwargs_ranges)
    return suggest_kwargs(
        trial=trial,
        prefix=prefix,
        kwargs_ranges=_kwargs_ranges,
        kwargs=kwargs,
    )


def suggest_discrete_uniform_int(
        trial: optuna.Trial,
        name, low, high, q) -> int:
    """

    Suggest an integer in the given range [low, high]
    inclusive with step size q.

    """
    if (high - low) % q:
        log.warning(
            f'bad range given: range({low}, {high}, {q}) -'
            f' not divisible by q')
    choices = list(range(low, high + 1, q))
    return trial.suggest_categorical(name=name, choices=choices)


def suggest_discrete_power_two_int(
        trial: optuna.Trial,
        name, low, high) -> int:
    """

    Suggest an integer in the given range [2^low, 2^high].

    """
    if high <= low:
        raise Exception(
            'Upper bound {high} is not greater than'
            f' lower bound {low}.')
    choices = [2 ** i for i in range(low, high + 1)]
    return trial.suggest_categorical(name=name, choices=choices)


def suggest_kwargs(
    trial: optuna.Trial,
    prefix: str,
    kwargs_ranges: Mapping[str, Any],
    kwargs: Optional[Mapping[str, Any]] = None,
):
    _kwargs = {}
    if kwargs:
        _kwargs.update(kwargs)

    for name, info in kwargs_ranges.items():
        if name in _kwargs:
            continue  # has been set by default, won't be suggested

        prefixed_name = f'{prefix}.{name}'
        dtype, low, high = info['type'], info.get('low'), info.get('high')
        if dtype in {int, 'int'}:
            q, scale = info.get('q'), info.get('scale')
            if scale == 'power_two':
                _kwargs[name] = suggest_discrete_power_two_int(
                    trial=trial,
                    name=prefixed_name,
                    low=low,
                    high=high,
                )
            elif q is not None:
                _kwargs[name] = suggest_discrete_uniform_int(
                    trial=trial,
                    name=prefixed_name,
                    low=low,
                    high=high,
                    q=q,
                )
            else:
                _kwargs[name] = trial.suggest_int(
                    name=prefixed_name,
                    low=low,
                    high=high)

        elif dtype in {float, 'float'}:
            if info.get('scale') == 'log':
                _kwargs[name] = trial.suggest_loguniform(
                    name=prefixed_name, low=low, high=high)
            else:
                _kwargs[name] = trial.suggest_uniform(
                    name=prefixed_name, low=low, high=high)
        elif dtype == 'categorical':
            choices = info['choices']
            _kwargs[name] = trial.suggest_categorical(
                name=prefixed_name, choices=choices)
        elif dtype in {bool, 'bool'}:
            _kwargs[name] = trial.suggest_categorical(
                name=prefixed_name, choices=[True, False])
        else:
            log.warning(f'Unhandled data type ({dtype}) for parameter {name}')

    return _kwargs


STOPPED_EPOCH_KEY = 'stopped_epoch'


@dataclass
class Objective:

    model: Type[keen_models_base.Model]  # 2.
    loss: Type[keen_losses.Loss]  # 3.
    regularizer: Type[keen_regs.Regularizer]  # 4.
    optimizer: Type[keen_optims.Optimizer]  # 5.
    training_loop: Type[keen_training.TrainingLoop]  # 6.
    evaluator: Type[keen_eval.Evaluator]  # 8.
    result_tracker: Type[keen_trackers.ResultTracker]  # 9.

    # 1. Dataset
    training_triples_factory: Optional[keen_triples.TriplesFactory] = None
    testing_triples_factory: Optional[keen_triples.TriplesFactory] = None
    validation_triples_factory: Optional[keen_triples.TriplesFactory] = None

    # 2. Model
    model_kwargs: Optional[Mapping[str, Any]] = None
    model_kwargs_ranges: Optional[Mapping[str, Any]] = None

    # 3. Loss
    loss_kwargs: Optional[Mapping[str, Any]] = None
    loss_kwargs_ranges: Optional[Mapping[str, Any]] = None

    # 4. Regularizer
    regularizer_kwargs: Optional[Mapping[str, Any]] = None
    regularizer_kwargs_ranges: Optional[Mapping[str, Any]] = None

    # 5. Optimizer
    optimizer_kwargs: Optional[Mapping[str, Any]] = None
    optimizer_kwargs_ranges: Optional[Mapping[str, Any]] = None

    # 6. Training Loop
    negative_sampler: Optional[Type[keen_sampling.NegativeSampler]] = None
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None
    negative_sampler_kwargs_ranges: Optional[Mapping[str, Any]] = None

    # 7. Training
    training_kwargs: Optional[Mapping[str, Any]] = None
    training_kwargs_ranges: Optional[Mapping[str, Any]] = None
    stopper: Type[keen_stoppers.Stopper] = None
    stopper_kwargs: Optional[Mapping[str, Any]] = None

    # 8. Evaluation
    evaluator_kwargs: Optional[Mapping[str, Any]] = None
    evaluation_kwargs: Optional[Mapping[str, Any]] = None

    # 9. Trackers
    result_tracker_kwargs: Optional[Mapping[str, Any]] = None

    # Misc.
    metric: str = None
    device: Union[None, str, torch.device] = None
    save_model_directory: Optional[str] = None

    @staticmethod
    def _update_stopper_callbacks(
            stopper_kwargs: Dict[str, Any],
            trial: optuna.Trial) -> None:

        """Make a subclass of the EarlyStopper that reports to the trial."""

        def _result_callback(
                _early_stopper: keen_stoppers.EarlyStopper,
                result: Union[float, int], epoch: int) -> None:
            trial.report(result, step=epoch)

        def _stopped_callback(
                _early_stopper: keen_stoppers.EarlyStopper,
                _result: Union[float, int], epoch: int) -> None:
            trial.set_user_attr(STOPPED_EPOCH_KEY, epoch)

        for key, callback in zip(
                ('result_callbacks', 'stopped_callbacks'),
                (_result_callback, _stopped_callback)):
            stopper_kwargs.setdefault(key, []).append(callback)

    def __call__(self, trial: optuna.Trial) -> Optional[float]:
        """Suggest parameters then train the model."""
        if self.model_kwargs is not None:
            problems = [
                x for x in (
                    'loss', 'regularizer', 'optimizer',
                    'training', 'negative_sampler', 'stopper')
                if x in self.model_kwargs
            ]
            if problems:
                raise ValueError(
                    f'model_kwargs should not have: {problems}. {self}')

        # 2. Model
        _model_kwargs = _get_kwargs(
            trial=trial,
            prefix='model',
            default_kwargs_ranges=self.model.hpo_default,
            kwargs=self.model_kwargs,
            kwargs_ranges=self.model_kwargs_ranges,
        )
        # 3. Loss
        _loss_kwargs = _get_kwargs(
            trial=trial,
            prefix='loss',
            default_kwargs_ranges=keen_losses.losses_hpo_defaults[self.loss],
            kwargs=self.loss_kwargs,
            kwargs_ranges=self.loss_kwargs_ranges,
        )
        # 4. Regularizer
        _regularizer_kwargs = _get_kwargs(
            trial=trial,
            prefix='regularizer',
            default_kwargs_ranges=self.regularizer.hpo_default,
            kwargs=self.regularizer_kwargs,
            kwargs_ranges=self.regularizer_kwargs_ranges,
        )
        # 5. Optimizer
        _optimizer_kwargs = _get_kwargs(
            trial=trial,
            prefix='optimizer',
            default_kwargs_ranges=keen_optims.optimizers_hpo_defaults[
                self.optimizer],
            kwargs=self.optimizer_kwargs,
            kwargs_ranges=self.optimizer_kwargs_ranges,
        )

        # 9. Tracker
        _exp_name = self.result_tracker_kwargs['experiment']
        _result_tracker_kwargs = {
            **self.result_tracker_kwargs,
            **dict(experiment=f'{_exp_name}-{trial.number}')}

        if self.training_loop is not keen_training.SLCWATrainingLoop:
            _negative_sampler_kwargs = {}
        else:
            _negative_sampler_kwargs = _get_kwargs(
                trial=trial,
                prefix='negative_sampler',
                default_kwargs_ranges=self.negative_sampler.hpo_default,
                kwargs=self.negative_sampler_kwargs,
                kwargs_ranges=self.negative_sampler_kwargs_ranges,
            )

        _training_kwargs = _get_kwargs(
            trial=trial,
            prefix='training',
            default_kwargs_ranges=self.training_loop.hpo_default,
            kwargs=self.training_kwargs,
            kwargs_ranges=self.training_kwargs_ranges,
        )

        _stopper_kwargs = dict(self.stopper_kwargs or {})
        if self.stopper is not None and issubclass(
                self.stopper, keen_stoppers.EarlyStopper):
            self._update_stopper_callbacks(_stopper_kwargs, trial)

        try:
            log.info('! intializing new pipeline')
            result = keen_pipeline.pipeline(
                # 1. Dataset
                training_triples_factory=self.training_triples_factory,
                validation_triples_factory=self.validation_triples_factory,
                testing_triples_factory=self.testing_triples_factory,

                # 2. Model
                model=self.model,
                model_kwargs=_model_kwargs,

                # 3. Loss
                loss=self.loss,
                loss_kwargs=_loss_kwargs,

                # 4. Regularizer
                regularizer=self.regularizer,
                regularizer_kwargs=_regularizer_kwargs,
                clear_optimizer=True,

                # 5. Optimizer
                optimizer=self.optimizer,
                optimizer_kwargs=_optimizer_kwargs,

                # 6. Training Loop
                training_loop=self.training_loop,
                negative_sampler=self.negative_sampler,
                negative_sampler_kwargs=_negative_sampler_kwargs,

                # 7. Training
                training_kwargs=_training_kwargs,
                stopper=self.stopper,
                stopper_kwargs=_stopper_kwargs,

                # 8. Evaluation
                evaluator=self.evaluator,
                evaluator_kwargs=self.evaluator_kwargs,
                evaluation_kwargs=self.evaluation_kwargs,

                # 9. Tracker
                result_tracker=self.result_tracker,
                result_tracker_kwargs=_result_tracker_kwargs,

                # Misc.
                use_testing_data=False,  # use validation set during HPO!
                device=self.device,
            )

        except (MemoryError, RuntimeError) as e:
            trial.set_user_attr('failure', str(e))
            # Will trigger Optuna to set the state of the trial as failed
            return None

        else:
            if self.save_model_directory:
                model_directory = os.path.join(
                    self.save_model_directory,
                    str(trial.number))

                os.makedirs(model_directory, exist_ok=True)
                result.save_to_directory(model_directory)

            trial.set_user_attr('random_seed', result.random_seed)

            for k, v in result.metric_results.to_flat_dict().items():
                trial.set_user_attr(k, v)

            return result.metric_results.get_metric(self.metric)


def hpo_pipeline(
    *,
    # 1. Dataset
    dataset: keen.Dataset,

    # 2. Model
    model: Union[str, Type[keen_models_base.Model]],
    model_kwargs: Optional[Mapping[str, Any]] = None,
    model_kwargs_ranges: Optional[Mapping[str, Any]] = None,

    # 3. Loss
    loss: Union[None, str] = None,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    loss_kwargs_ranges: Optional[Mapping[str, Any]] = None,

    # 4. Regularizer
    regularizer: Union[None, str] = None,
    regularizer_kwargs: Optional[Mapping[str, Any]] = None,
    regularizer_kwargs_ranges: Optional[Mapping[str, Any]] = None,

    # 5. Optimizer
    optimizer: Union[None, str] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    optimizer_kwargs_ranges: Optional[Mapping[str, Any]] = None,

    # 6. Training Loop
    training_loop: Union[None, str] = None,
    negative_sampler: Union[None, str] = None,
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
    negative_sampler_kwargs_ranges: Optional[Mapping[str, Any]] = None,

    # 7. Training
    training_kwargs: Optional[Mapping[str, Any]] = None,
    training_kwargs_ranges: Optional[Mapping[str, Any]] = None,
    stopper: Union[None, str] = None,
    stopper_kwargs: Optional[Mapping[str, Any]] = None,

    # 8. Evaluation
    evaluator: Union[None, str] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    metric: Optional[str] = None,

    # 9. Tracking
    result_tracker: Union[None, str, Type[keen_trackers.ResultTracker]] = None,
    result_tracker_kwargs: Optional[Mapping[str, Any]] = None,

    # 6. Misc
    device: Union[None, str, torch.device] = None,

    #  Optuna Study Settings
    storage: Union[None, str, BaseStorage] = None,
    sampler: Union[None, str, Type[BaseSampler]] = None,
    sampler_kwargs: Optional[Mapping[str, Any]] = None,
    pruner: Union[None, str, Type[BasePruner]] = None,
    pruner_kwargs: Optional[Mapping[str, Any]] = None,
    study_name: Optional[str] = None,
    direction: Optional[str] = None,
    load_if_exists: bool = False,

    # Optuna Optimization Settings
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    n_jobs: Optional[int] = None,
    save_model_directory: Optional[str] = None,

) -> hpo.HpoPipelineResult:
    """
    Train a model on the given dataset.

    See original documentation at pykeen for more details
    """
    log.info('running ')

    sampler_cls = hpo_samplers.get_sampler_cls(sampler)
    pruner_cls = hpo_pruners.get_pruner_cls(pruner)

    if direction is None:
        direction = 'minimize'

    study = optuna.create_study(
        storage=storage,
        sampler=sampler_cls(**(sampler_kwargs or {})),
        pruner=pruner_cls(**(pruner_kwargs or {})),
        study_name=study_name,
        direction=direction,
        load_if_exists=load_if_exists,
    )

    # 0. Metadata/Provenance
    study.set_user_attr('pykeen_version', keen_version.get_version())
    study.set_user_attr('pykeen_git_hash', keen_version.get_git_hash())

    # 1. Dataset
    study.set_user_attr('dataset', dataset.name)

    # 2. Model
    model = keen_models.get_model_cls(model)
    study.set_user_attr(
        'model',
        keen_utils.normalize_string(model.__name__))

    log.info(f'using model: {model}')

    # 3. Loss
    loss: Type[keen_losses.Loss] = (
        model.loss_default
        if loss is None
        else keen_losses.get_loss_cls(loss))

    study.set_user_attr(
        'loss',
        keen_utils.normalize_string(
            loss.__name__,
            suffix=keen_losses._LOSS_SUFFIX))

    log.info(f'using loss: {loss}')

    # 4. Regularizer
    regularizer = (
        model.regularizer_default if not regularizer else
        keen_regs.get_regularizer_cls(regularizer))

    study.set_user_attr(
        'regularizer',
        regularizer.get_normalized_name())

    log.info(f'using regularizer: {regularizer}')

    # 5. Optimizer
    optimizer = keen_optims.get_optimizer_cls(optimizer)
    study.set_user_attr(
        'optimizer',
        keen_utils.normalize_string(optimizer.__name__))

    log.info(f'using optimizer: {optimizer}')

    # 6. Training Loop
    training_loop = keen_training.get_training_loop_cls(training_loop)

    study.set_user_attr(
        'training_loop',
        training_loop.get_normalized_name())

    log.info(f'using training loop: {training_loop}')

    negative_sampler = None
    if training_loop is keen_training.SLCWATrainingLoop:
        negative_sampler = keen_sampling.get_negative_sampler_cls(
            negative_sampler)

        study.set_user_attr(
            'negative_sampler',
            negative_sampler.get_normalized_name())

        log.info(f'using negative sampler: {negative_sampler}')

    # 7. Training
    stopper = keen_stoppers.get_stopper_cls(stopper)

    if (
            stopper is keen_stoppers.EarlyStopper
            and training_kwargs_ranges
            and 'epochs' in training_kwargs_ranges):

        raise ValueError('can not use early stopping while optimizing epochs')

    # 8. Evaluation
    evaluator = keen_eval.get_evaluator_cls(evaluator)
    study.set_user_attr('evaluator', evaluator.get_normalized_name())
    log.info(f'using evaluator: {evaluator}')

    if metric is None:
        metric = 'adjusted_mean_rank'
        log.info(f'! using default metric: "{metric}"')

    study.set_user_attr('metric', metric)
    log.info(f'attempting to {direction} {metric}')

    # 9. Tracking
    result_tracker = keen_trackers.get_result_tracker_cls(result_tracker)

    objective = Objective(
        # 1. Dataset
        training_triples_factory=dataset.training,
        validation_triples_factory=dataset.validation,
        testing_triples_factory=dataset.testing,

        # 2. Model
        model=model,
        model_kwargs=model_kwargs,
        model_kwargs_ranges=model_kwargs_ranges,

        # 3. Loss
        loss=loss,
        loss_kwargs=loss_kwargs,
        loss_kwargs_ranges=loss_kwargs_ranges,

        # 4. Regularizer
        regularizer=regularizer,
        regularizer_kwargs=regularizer_kwargs,
        regularizer_kwargs_ranges=regularizer_kwargs_ranges,

        # 5. Optimizer
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_kwargs_ranges=optimizer_kwargs_ranges,

        # 6. Training Loop
        training_loop=training_loop,
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
        negative_sampler_kwargs_ranges=negative_sampler_kwargs_ranges,

        # 7. Training
        training_kwargs=training_kwargs,
        training_kwargs_ranges=training_kwargs_ranges,
        stopper=stopper,
        stopper_kwargs=stopper_kwargs,

        # 8. Evaluation
        evaluator=evaluator,
        evaluator_kwargs=evaluator_kwargs,
        evaluation_kwargs=evaluation_kwargs,

        # 9. Tracker
        result_tracker=result_tracker,
        result_tracker_kwargs=result_tracker_kwargs,

        # Optuna Misc.
        metric=metric,
        save_model_directory=save_model_directory,

        # Pipeline Misc.
        device=device,
    )

    # Invoke optimization of the objective function.
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs or 1,
    )

    return hpo.HpoPipelineResult(
        study=study,
        objective=objective,
    )
