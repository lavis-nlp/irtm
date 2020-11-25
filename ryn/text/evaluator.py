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
        datamodule: data.DataModule = None,
        debug: bool = None,
):
    print('''

              R Y N
    -------------------------
            evaluation

    ''')

    print('producing projections\n')

    hvd.init()
    assert hvd.local_rank() == 0, 'no multi gpu support so far'

    model.init_projections()
    model.run_memcheck(test=True)
    model.debug = debug
    model.eval()

    loaders = (
        [datamodule.train_dataloader()] +
        datamodule.val_dataloader() +
        [datamodule.test_dataloader()]
    )

    print('\nrunning kgc evaluation\n')

    tqdm = partial(_tqdm, ncols=80, unit='batches')
    with torch.no_grad():
        for loader in loaders:
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

    triplesens = {
        'transductive': datamodule.kgc.transductive,
        'inductive': datamodule.kgc.inductive,
        'test': datamodule.kgc.test,
    }

    results = {}
    for kind, triples in triplesens.items():
        results[kind] = model.run_kgc_evaluation(
            kind=kind,
            triples=triples
        )

    return results


@helper.notnone
def _evaluation_uncached(
        out: pathlib.Path = None,
        config: List[str] = None,
        checkpoint: pathlib.Path = None,
        debug: bool = None,
):

    config = Config.create(configs=[out / 'config.yml'] + list(config))
    datamodule, rync = trainer.load_from_config(config=config)

    datamodule.prepare_data()
    datamodule.setup('test')

    model = mapper.Mapper.load_from_checkpoint(
        str(checkpoint),
        data=datamodule,
        rync=rync,
        freeze_text_encoder=config.freeze_text_encoder
    )

    model = model.to(device='cuda')

    results = evaluate(
        model=model,
        datamodule=datamodule,
        debug=debug,
    )

    return results


@helper.notnone
@helper.cached('.cached.text.evaluator.result.{checkpoint_name}.pkl')
def _evaluation_cached(
        *,
        path: pathlib.Path = None,
        checkpoint_name: str = None,
        **kwargs,
):
    return _evaluation_uncached(**kwargs)


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

    ryn_dir = path / 'ryn'
    helper.path(ryn_dir, create=True)

    if not debug:
        results = _evaluation_cached(
            # helper.cached
            path=ryn_dir,
            checkpoint_name=checkpoint.name,
            # evaluate
            out=path,
            config=config,
            checkpoint=checkpoint,
            debug=debug,
        )
    else:
        results = _evaluation_uncached(
            # evaluate
            out=path,
            config=config,
            checkpoint=checkpoint,
            debug=debug,
        )

    yml_path = ryn_dir / 'evaluation.yml'
    yml_str = yaml.dump(results)

    if not debug:
        with yml_path.open(mode='w') as fd:
            fd.write(yml_str)

    print('\n\nfinished!\n')
    print(yml_str)
