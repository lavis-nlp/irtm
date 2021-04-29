# -*- coding: utf-8 -*-

import irtm
from irtm.common import ryaml

from irtm.text import util
from irtm.text import data
from irtm.text import mapper
from irtm.text import trainer
from irtm.text.config import Config
from irtm.common import helper

import csv
import torch
import logging
from tqdm import tqdm as _tqdm

import pathlib
from functools import partial
from dataclasses import replace

from typing import List
from typing import Dict
from typing import Union


log = logging.getLogger(__name__)


def _init(
    model: mapper.Mapper,
    debug: bool,
):
    model = model.to(device="cuda")
    model.debug = debug
    model.eval()


def _create_projections(
    model: mapper.Mapper,
    datamodule: data.DataModule,
    debug: bool,
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


def _run_kgc_evaluations(
    model: mapper.Mapper,
    datamodule: data.DataModule,
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


def evaluate(
    model: mapper.Mapper,
    datamodule: data.DataModule,
    debug: bool,
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


def _evaluation_uncached(
    out: pathlib.Path,
    config: List[str],
    checkpoint: pathlib.Path,
    debug: bool,
):

    config = Config.create(configs=[out / "config.yml"] + list(config))
    config = replace(config, split_text_dataset=False)

    datamodule, irtmc = trainer.load_from_config(config=config)

    datamodule.prepare_data()
    datamodule.setup("test")

    model = mapper.Mapper.load_from_checkpoint(
        str(checkpoint),
        data=datamodule,
        irtmc=irtmc,
        freeze_text_encoder=True,
    )

    results = evaluate(
        model=model,
        datamodule=datamodule,
        debug=debug,
    )

    return results


@helper.cached(".cached.text.evaluator.result.{checkpoint_name}.pkl")
def _evaluation_cached(
    path: pathlib.Path,
    checkpoint_name: str,
    **kwargs,
):
    return _evaluation_uncached(**kwargs)


def _handle_results(
    results: Dict,
    checkpoint: str,
    target_file: Union[str, pathlib.Path],
    debug: bool,
):
    if not debug:
        runs = {}

        try:
            runs = ryaml.load([target_file]) or {}
        except FileNotFoundError:
            log.info(f"creating {target_file.name}")

        runs[checkpoint] = results
        with target_file.open(mode="w") as fd:
            fd.write(yaml.dump(runs))

    else:
        print("\n\n")
        print(yaml.dump(results))


def evaluate_from_kwargs(
    path: Union[pathlib.Path, str],
    checkpoint: Union[pathlib.Path, str],
    config: List[str],
    debug: bool,
):

    path = helper.path(path, exists=True, message="loading data from {path_abbrv}")

    checkpoint = helper.path(
        checkpoint, exists=True, message="loading checkpoint from {path_abbrv}"
    )

    print(f"evaluating {checkpoint.name}")
    irtm_dir = path / "irtm"

    if not debug:
        helper.path(irtm_dir, create=True)

        results = _evaluation_cached(
            # helper.cached
            path=irtm_dir,
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
        target_file=irtm_dir / f"evaluation.{checkpoint.name}.yml",
        debug=debug,
    )

    return checkpoint.name, results


def evaluate_baseline(
    config: List[str],
    out: str,
    debug: bool,
    **kwargs,
):
    config = Config.create(configs=config, **kwargs)
    datamodule, irtmc = trainer.load_from_config(config=config)

    model = mapper.Mapper(
        irtmc=irtmc,
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

    out = helper.path(out, create=True, message="writing results to {path_abbrv}")

    _handle_results(
        results=results,
        target_file=out / "evaluation.baseline.yml",
        debug=debug,
    )


def evaluate_all(root: Union[str, pathlib.Path], **kwargs):
    """
    Run evaluation for all saved checkpoints
    """

    root = helper.path(root, exists=True)
    for checkpoint in root.glob("**/epoch=*-step=*.ckpt"):
        # <path>/weights/<PROJECT_NAME>/<RUN_ID>/checkpoints/epoch=*-step=*.ckpt
        path = checkpoint.parents[4]
        evaluate_from_kwargs(path=path, checkpoint=checkpoint, **kwargs)


def evaluate_csv(
    csv_file: Union[str, pathlib.Path],
    debug: bool,
    only_marked: bool,
    **kwargs,
):
    """
    Run evaluations based on a csv file
    """

    results = []
    exps = util.Experiments(csv_file)

    for exp in exps:
        if not exp.name:
            print(f"skipping {exp} (no name)")
            continue

        if only_marked and exp.note != "EVAL":
            print(f"skipping {exp} (not marked EVAL)")

        try:
            path, checkpoint = exp.path, exp.path_checkpoint

        except irtm.IRTMError as exc:
            print(str(exc))
            continue

        print(f"\n{exp}")
        _, ret = evaluate_from_kwargs(
            path=path,
            checkpoint=checkpoint,
            debug=debug,
            **kwargs,
        )

        results.append((exp, ret))

    if debug:
        return

    out_file = exps.path.parent / (exps.path.name + ".results.csv")
    with out_file.open(mode="w") as fd:
        writer = csv.writer(fd)
        writer.writerows(
            [
                str(exp),
                res["test"]["hits_at_k"]["both"]["avg"][1],
                res["test"]["hits_at_k"]["both"]["avg"][5],
                res["test"]["hits_at_k"]["both"]["avg"][10],
                res["test"]["mean_reciprocal_rank"]["both"]["avg"],
                res["inductive"]["hits_at_k"]["both"]["avg"][10],
                res["inductive"]["mean_reciprocal_rank"]["both"]["avg"],
            ]
            for exp, res in results
        )
