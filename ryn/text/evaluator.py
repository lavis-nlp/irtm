# -*- coding: utf-8 -*-

from ryn.text import data
from ryn.text import mapper
from ryn.text import trainer
from ryn.text.config import Config
from ryn.common import helper
from ryn.common import logging

import yaml
import torch
import horovod.torch as hvd
from tqdm import tqdm as _tqdm

import pathlib
from functools import partial

from typing import List
from typing import Union


log = logging.get('text.evaluator')


@helper.notnone
def evaluate(
        *,
        model: mapper.Mapper = None,
        datasets: data.Datasets = None,
        out: Union[str, pathlib.Path] = None,
        debug: bool = None,
):
    print('''

              R Y N
    -------------------------
            evaluation

    ''')

    print('producing projections\n')

    hvd.init()

    model.init_projections()
    model.run_memcheck(test=True)
    model.debug = debug
    model.eval()

    work = {
        'transductive': (
            datasets.text_train,
            datasets.kgc_transductive),
        'inductive': (
            datasets.text_inductive,
            datasets.kgc_inductive,
        ),
        'test': (
            datasets.text_test,
            datasets.kgc_test,
        )
    }

    tqdm = partial(_tqdm, ncols=80, unit='batches')

    print('\nrunning kgc evaluation\n')

    with torch.no_grad():
        for loader, _ in work.values():
            gen = tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f'{loader.dataset.name} samples ',
            )

            for batch_idx, batch in gen:
                sentences, entities = batch
                sentences = sentences.to(device=model.device)
                projected = model.forward(sentences=sentences)

                model.update_projections(
                    entities=entities,
                    projected=projected
                )

                if debug:
                    break

    results = {}
    for kind, (_, triples) in work.items():
        results[kind] = model.run_kgc_evaluation(
            kind=kind,
            triples=triples
        )

    out = helper.path(out, message='write results to {path_abbrv}')
    helper.path(out.parent, create=True)
    yamlized = yaml.dump(results)

    if not debug:
        with out.open(mode='w') as fd:
            fd.write(yamlized)

    print('\n\nfinished! uwu\n')
    print(yamlized)


@helper.notnone
def evaluate_from_kwargs(
        *,
        path: Union[pathlib.Path, str] = None,
        checkpoint: Union[pathlib.Path, str] = None,
        config: List[str] = None,
        debug: bool = None,
):
    path = helper.path(
        path, exists=True,
        message='loading data from {path_abbrv}')

    checkpoint = helper.path(
        checkpoint, exists=True,
        message='loading checkpoint from {path_abbrv}')

    config = Config.create(configs=[path / 'config.yml'] + list(config))
    datasets, rync = trainer.load_from_config(config=config)

    model = mapper.Mapper.load_from_checkpoint(
        str(checkpoint),
        datasets=datasets,
        rync=rync,
        freeze_text_encoder=config.freeze_text_encoder
    )

    model = model.to(device='cuda')
    evaluate(
        model=model,
        datasets=datasets,
        out=path/'evaluation.yml',
        debug=debug
    )
