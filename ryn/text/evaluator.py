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
from typing import Dict
from typing import Union


log = logging.get("text.evaluator")


@helper.notnone
def _init(
    *,
    model: mapper.Mapper = None,
    run_memcheck: bool = None,
    debug: bool = None,
):
    model = model.to(device="cuda")

    hvd.init()
    assert hvd.local_rank() == 0, "no multi gpu support so far"

    if run_memcheck:
        model.run_memcheck(test=True)

    model.init_projections()
    model.debug = debug
    model.eval()


@helper.notnone
def _create_projections(
    *,
    model: mapper.Mapper = None,
    datamodule: data.DataModule = None,
    debug: bool = None,
):
    loaders = (
        [datamodule.train_dataloader()]
        + datamodule.val_dataloader()
        + [datamodule.test_dataloader()]
    )

    tqdm = partial(_tqdm, ncols=80, unit="batches")
    with torch.no_grad():
        for loader in loaders:
            gen = tqdm(
                enumerate(loader),
                total=len(loader),
                desc=f"{loader.dataset.name} samples ",
            )

            for batch_idx, batch in gen:
                sentences, entities = batch
                sentences = sentences.to(device=model.device)
                projected = model.forward(sentences=sentences)

                model.update_projections(
                    entities=entities, projected=projected
                )

                if debug:
                    break


@helper.notnone
def _run_kgc_evaluations(
    *,
    model: mapper.Mapper = None,
    datamodule: data.DataModule = None,
):
    triplesens = {
        "transductive": datamodule.kgc.transductive,
        "inductive": datamodule.kgc.inductive,
        "test": datamodule.kgc.test,
    }

    results = {}

    for kind, triples in triplesens.items():
        key = f"{model.projection_aggregation}.{kind}"
        results[key] = model.run_kgc_evaluation(
            kind=kind,
            triples=triples,
        )

    return results


@helper.notnone
def evaluate(
    *,
    model: mapper.Mapper = None,
    datamodule: data.DataModule = None,
    debug: bool = None,
):
    print("EVALUATE")

    _init(model=model, debug=debug, run_memcheck=True)

    results = {}
    projection_aggregations = "max", "avg", "last"
    for projection_aggregation in projection_aggregations:

        print("\ncreating projections\n")
        _create_projections(
            model=model,
            datamodule=datamodule,
            debug=debug,
        )

        print("\nrunning kgc evaluation\n")
        results.update(
            _run_kgc_evaluations(model=model, datamodule=datamodule)
        )

    return results


@helper.notnone
def _evaluation_uncached(
    out: pathlib.Path = None,
    config: List[str] = None,
    checkpoint: pathlib.Path = None,
    debug: bool = None,
):

    config = Config.create(configs=[out / "config.yml"] + list(config))
    datamodule, rync = trainer.load_from_config(config=config)

    datamodule.prepare_data()
    datamodule.setup("test")

    model = mapper.Mapper.load_from_checkpoint(
        str(checkpoint),
        data=datamodule,
        rync=rync,
        freeze_text_encoder=True,
    )

    results = evaluate(
        model=model,
        datamodule=datamodule,
        debug=debug,
    )

    return results


@helper.notnone
@helper.cached(".cached.text.evaluator.result.{checkpoint_name}.pkl")
def _evaluation_cached(
    *,
    path: pathlib.Path = None,
    checkpoint_name: str = None,
    **kwargs,
):
    return _evaluation_uncached(**kwargs)


@helper.notnone
def _handle_results(
    *,
    results: Dict,
    checkpoint: str = None,
    target_file: Union[str, pathlib.Path],
    debug: bool = None,
):
    if not debug:
        with target_file.open(mode="a") as fd:
            runs = yaml.load(fd) or {}

        runs[checkpoint] = results
        with target_file.open(mode="w") as fd:
            fd.write(yaml.dump(runs))

    print("\n\nfinished!\n")
    print(yaml.dump(results))


@helper.notnone
def evaluate_from_kwargs(
    *,
    path: Union[pathlib.Path, str] = None,
    checkpoint: Union[pathlib.Path, str] = None,
    config: List[str] = None,
    debug: bool = None,
):

    path = helper.path(
        path, exists=True, message="loading data from {path_abbrv}"
    )

    checkpoint = helper.path(
        checkpoint, exists=True, message="loading checkpoint from {path_abbrv}"
    )

    ryn_dir = path / "ryn"
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

    _handle_results(
        checkpoint=checkpoint.name,
        results=results,
        target_file=ryn_dir / "evaluation.yml",
        debug=debug,
    )

    return checkpoint.name, results


@helper.notnone
def evaluate_baseline(
    *,
    config: List[str] = None,
    out: str = None,
    debug: bool = None,
    **kwargs,
):
    config = Config.create(configs=config, **kwargs)
    datamodule, rync = trainer.load_from_config(config=config)

    model = mapper.Mapper(
        rync=rync,
        data=datamodule,
        freeze_text_encoder=True,
    )

    _init(model=model, debug=debug, run_memcheck=False)

    # control experiment that there's no test leakage
    model.debug = debug
    model.projections.fill_(1.0)
    model.projections_counts.fill_(1.0)

    results = _run_kgc_evaluations(
        model=model,
        datamodule=datamodule,
    )

    out = helper.path(
        out, create=True, message="writing results to {path_abbrv}"
    )

    _handle_results(
        results=results,
        target_file=out / "evaluation.yml",
        debug=debug,
    )


@helper.notnone
def evaluate_all(root: pathlib.Path = None, **kwargs):
    """
    Run evaluation for all saved checkpoints
    """

    root = helper.path(root, exists=True)
    for checkpoint in root.glob("**/epoch=*-step=*.ckpt"):
        print(f"evaluating {checkpoint.name}")

        # <path>/weights/<PROJECT_NAME>/<RUN_ID>/checkpoints/epoch=*-step=*.ckpt
        path = checkpoint.parents[4]
        evaluate_from_kwargs(path=path, checkpoint=checkpoint, **kwargs)
