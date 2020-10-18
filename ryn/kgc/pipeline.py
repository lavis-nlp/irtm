# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import config
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

import torch
import numpy as np

import dataclasses

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
def single(
        *,
        config: config.Config = None,
        split_dataset: split.Dataset = None,
        keen_dataset: keen.Dataset = None,
) -> None:
    # TODO return value
    # TODO introduce regularizers

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
        config.loss)

    regularizer = config.resolve(
        config.regularizer,
        device=device)

    model = config.resolve(
        config.model,
        loss=loss,
        regularizer=regularizer,
        random_seed=config.general.seed,
        triples_factory=keen_dataset.training,
        preferred_device=device,)

    optimizer = config.resolve(
        config.optimizer,
        params=model.get_grad_params())

    evaluator = config.resolve(
        config.evaluator)

    stopper = config.resolve(
        config.stopper,
        model=model,
        evaluator=evaluator,
        evaluation_triples_factory=keen_dataset.training,
        result_tracker=result_tracker,
        evaluation_batch_size=config.evaluator.batch_size)

    training_loop = config.resolve(
        config.training_loop,
        model=model,
        optimizer=optimizer,
        negative_sampler_cls=config.sampler.constructor,
        negative_sampler_kwargs=config.sampler.kwargs)

    # kindling

    # losses = training_loop(...
    training_loop.train(**{
        **dataclasses.asdict(config.training),
        **dict(
            stopper=stopper,
            result_tracker=result_tracker,
            # TODO work out how to resume from checkpoint
            clear_optimizer=False,
        )
    })
