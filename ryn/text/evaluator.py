# -*- coding: utf-8 -*-

import ryn
from ryn.text import data
from ryn.text import mapper
from ryn.text import trainer
from ryn.text.config import Config
from ryn.common import helper
from ryn.common import logging

import csv
import yaml
import torch
import horovod.torch as hvd
from tqdm import tqdm as _tqdm

import pathlib
from functools import partial
from dataclasses import replace

from typing import List
from typing import Dict
from typing import Union


log = logging.get("text.evaluator")


@helper.notnone
def _init(
    *,
    model: mapper.Mapper = None,
    debug: bool = None,
):
    model = model.to(device="cuda")

    hvd.init()
    assert hvd.local_rank() == 0, "no multi gpu support so far"

    model.debug = debug
    model.eval()


@helper.notnone
def _create_projections(
    *,
    model: mapper.Mapper = None,
    datamodule: data.DataModule = None,
    debug: bool = None,
):

    model.init_projections()

    loaders = (
        [datamodule.train_dataloader()]
        + datamodule.val_dataloader()
        + [datamodule.test_dataloader()]
    )

    tqdm = partial(_tqdm, ncols=80, unit="batches")
    with torch.no_grad():
        for loader in loaders:

            gen = ((b[0], b[1].to(model.device)) for b in loader)
            gen = tqdm(
                enumerate(gen),
                total=len(loader),
                desc=f"{loader.dataset.name} samples ",
            )

            for batch_idx, batch in gen:
                model.forward(batch=batch, subbatch_size=loader.subbatch_size)
                if debug:
                    break

    model.gather_projections()


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

    ret = {}
    for kind, triples in triplesens.items():
        results = model.run_kgc_evaluation(
            kind=kind,
            triples=triples,
        )

        ret[kind] = results.metrics

    return ret


@helper.notnone
def evaluate(
    *,
    model: mapper.Mapper = None,
    datamodule: data.DataModule = None,
    debug: bool = None,
):
    _init(model=model, debug=debug)

    print("\ncreating projections\n")
    _create_projections(
        model=model,
        datamodule=datamodule,
        debug=debug,
    )

    print("\nrunning kgc evaluation\n")
    results = _run_kgc_evaluations(
        model=model,
        datamodule=datamodule,
    )

    def _map(dic):
        mapped = {}
        for k, v in dic.items():
            if type(v) is dict:
                mapped[k] = _map(v)
            else:
                try:
                    # it is full of numpy scalars
                    # that yaml chokes on
                    mapped[k] = v.item()
                except AttributeError:
                    mapped[k] = v

        return mapped

    mapped = _map(results)
    return mapped


@helper.notnone
def _evaluation_uncached(
    out: pathlib.Path = None,
    config: List[str] = None,
    checkpoint: pathlib.Path = None,
    debug: bool = None,
):

    config = Config.create(configs=[out / "config.yml"] + list(config))
    config = replace(config, split_text_dataset=False)

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
    results: Dict = None,
    checkpoint: str = None,
    target_file: Union[str, pathlib.Path],
    debug: bool = None,
):
    if not debug:
        runs = {}

        try:
            with target_file.open(mode="r") as fd:
                runs = yaml.load(fd, Loader=yaml.FullLoader) or {}
        except FileNotFoundError:
            log.info(f"creating {target_file.name}")

        runs[checkpoint] = results
        with target_file.open(mode="w") as fd:
            fd.write(yaml.dump(runs))

    else:
        print("\n\n")
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

    print(f"evaluating {checkpoint.name}")
    ryn_dir = path / "ryn"

    if not debug:
        helper.path(ryn_dir, create=True)

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
        results=results,
        checkpoint=checkpoint.name,
        target_file=ryn_dir / f"evaluation.{checkpoint.name}.yml",
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

    _init(model=model, debug=debug)

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
        target_file=out / "evaluation.baseline.yml",
        debug=debug,
    )


@helper.notnone
def evaluate_all(root: Union[str, pathlib.Path] = None, **kwargs):
    """
    Run evaluation for all saved checkpoints
    """

    root = helper.path(root, exists=True)
    for checkpoint in root.glob("**/epoch=*-step=*.ckpt"):
        # <path>/weights/<PROJECT_NAME>/<RUN_ID>/checkpoints/epoch=*-step=*.ckpt
        path = checkpoint.parents[4]
        evaluate_from_kwargs(path=path, checkpoint=checkpoint, **kwargs)


def _csv_get_paths(row):
    for key in (
        "split",
        "text",
        "text-model",
        "kgc-model",
        "name",
        "run",
        "checkpoint",
    ):
        if key not in row or not row[key]:
            raise ryn.RynError(f"{row['name']} missing value in column: {key}")

    experiment_dir = helper.path(
        ryn.ENV.TEXT_DIR
        / "mapper"
        / row["split"]
        / row["text"]
        / row["text-model"]
        / row["kgc-model"],
        exists=True,
    )

    checkpoint = helper.path(
        experiment_dir
        / row["name"]
        / "weights"
        / "ryn-text"
        / row["run"]
        / "checkpoints"
        / row["checkpoint"],
        exists=True,
    )

    return experiment_dir / row["name"], checkpoint


@helper.notnone
def evaluate_csv(
    csv_file: Union[str, pathlib.Path] = None,
    debug: bool = None,
    **kwargs,
):
    """
    Run evaluations based on a csv file
    """

    def shorthand(row):
        trail = ".".join(
            row[k].strip() for k in ("identifier", "#sents", "mode")
        )
        name = f"{row['exp']} [{trail}]"
        if row["name"]:
            name += f" {row['name']}"

        return name

    csv_file = helper.path(
        csv_file, exists=True, message="loading csv data from {path_abbrv}"
    )

    with csv_file.open(mode="r") as fd:
        reader = csv.DictReader(fd)
        results = []

        for row in reader:
            if not row["name"]:
                print(f"skipping {shorthand(row)}")
                continue

            try:
                path, checkpoint = _csv_get_paths(row)

            except ryn.RynError as exc:
                print(str(exc))
                continue

            print(f"\n{shorthand(row)}")
            _, ret = evaluate_from_kwargs(
                path=path,
                checkpoint=checkpoint,
                debug=debug,
                **kwargs,
            )

            results.append((row, ret))

    if debug:
        return

    out_file = csv_file.parent / (csv_file.name + ".results.csv")
    with out_file.open(mode="w") as fd:
        writer = csv.writer(fd)
        writer.writerows(
            [
                shorthand(row),
                res["test"]["hits_at_k"]["both"]["avg"][1],
                res["test"]["hits_at_k"]["both"]["avg"][5],
                res["test"]["hits_at_k"]["both"]["avg"][10],
                res["test"]["mean_reciprocal_rank"]["both"]["avg"],
                res["inductive"]["hits_at_k"]["both"]["avg"][10],
                res["inductive"]["mean_reciprocal_rank"]["both"]["avg"],
            ]
            for row, res in results
        )
