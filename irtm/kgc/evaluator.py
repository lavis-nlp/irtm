import irt
import irtm

from irtm.kgc import data
from irtm.kgc.config import Config
from irtm.common import helper

import torch
from tabulate import tabulate
from tqdm import tqdm as _tqdm
from pykeen import evaluation as pk_evaluation

import csv
import logging
import pathlib
from functools import partial
from datetime import datetime

from typing import List
from typing import Union
from typing import Optional
from typing import Iterable


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


def evaluate(
    model: torch.nn.Module,
    config: Config,
    triples,
    tqdm_kwargs,
    filtered_triples: Optional[List] = None,
    **kwargs,
):
    tqdm_kwargs = tqdm_kwargs or {}
    filtered = filtered_triples is not None

    log.info(f"creating new evaluator: {filtered=}")
    evaluator = pk_evaluation.evaluator_resolver.make(
        config.evaluator.cls,
        automatic_memory_optimization=True,
        filtered=filtered,
    )

    ts = datetime.now()
    model = model.to("cuda")  # TODO (inter-machine problem)

    log.info("hand-off to pykeen for evaluation")
    metrics = evaluator.evaluate(
        model=model,
        mapped_triples=triples,
        tqdm_kwargs=tqdm_kwargs,
        additional_filter_triples=filtered_triples,
    )

    evaluation_time = data.Time(start=ts, end=datetime.now())

    evaluation_result = data.EvaluationResult(
        model=config.model.cls,
        created=datetime.now(),
        evaluation_time=evaluation_time,
        git_hash=helper.git_hash(),
        metrics=metrics.to_flat_dict(),
    )

    log.info(f"evaluation took: {evaluation_time.took}")
    return evaluation_result


def evaluate_glob(
    glob: Iterable[pathlib.Path],
    dataset: irt.Dataset,
    mode: str,
):

    glob = list(glob)
    log.info(f"probing {len(glob)} directories")

    results = []
    for path in tqdm(glob):
        try:
            training_result = data.TrainingResult.load(path)
            config = Config.load(path)

            assert config.general.seed

            kcw = irt.KeenClosedWorld(
                dataset=dataset,
                seed=config.general.seed,
                split=config.general.split,
            )

            assert training_result.config.general.dataset == kcw.dataset.name

        except (FileNotFoundError, NotADirectoryError, irtm.IRTMError) as exc:
            log.info(f"skipping {path.name}: {exc}")
            continue

        if path.is_file():
            log.info("loading evaluation result from file")
            eval_result = data.EvaluationResult.load(path, prefix=mode)

        else:
            if mode == "validation":
                triples = kcw.validation.mapped_triples
                filtered_triples = [kcw.training.mapped_triples]
            elif mode == "test":
                triples = kcw.testing.mapped_triples
                filtered_triples = [
                    kcw.training.mapped_triples,
                    kcw.validation.mapped_triples,
                ]

            log.info(f"evaluting {len(triples)} triples")

            eval_result = evaluate(
                model=training_result.model,
                config=training_result.config,
                triples=triples,
                filtered_triples=filtered_triples,
                tqdm_kwargs=dict(
                    position=1,
                    ncols=80,
                    leave=False,
                ),
            )

            eval_result.save(path, prefix=mode)

        results.append((eval_result, path))

    return results


def evaluate_from_kwargs(
    results: List[str],
    dataset: str,
    mode: str,
):

    dataset = irt.Dataset(dataset)

    eval_results = evaluate_glob(
        glob=map(pathlib.Path, results),
        dataset=dataset,
        mode=mode,
    )

    return eval_results


def print_results(
    results,
    out: Union[str, pathlib.Path],
    mode: str,
):
    rows = []
    sort_key = 4  # hits@10

    headers = [
        "name",
        "model",
        "val",
        "hits@1",
        "hits@10",
        "MRR",
    ]
    headers[sort_key] += " *"

    for eval_result, path in results:
        training_result = data.TrainingResult.load(path, load_model=False)

        rows.append(
            [
                path.name,
                eval_result.model,
                training_result.best_metric,
                eval_result.metrics["both.realistic.hits_at_1"] * 100,
                eval_result.metrics["both.realistic.hits_at_10"] * 100,
                eval_result.metrics["both.realistic.inverse_harmonic_mean_rank"],
            ]
        )

    rows.sort(key=lambda r: r[sort_key], reverse=True)
    table = tabulate(rows, headers=headers)

    print()
    print(table)

    if out is not None:
        fname = f"evaluation.{mode}"
        out = helper.path(
            out,
            create=True,
            message=f"writing {fname} txt/csv to {{path_abbrv}}",
        )

        with (out / (fname + ".txt")).open(mode="w") as fd:
            fd.write(table)

        with (out / (fname + ".csv")).open(mode="w") as fd:
            writer = csv.DictWriter(fd, fieldnames=headers)
            writer.writeheader()
            writer.writerows([dict(zip(headers, row)) for row in rows])
