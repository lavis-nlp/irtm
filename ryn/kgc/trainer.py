# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import data
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
import pathlib
import dataclasses
from functools import partial
from datetime import datetime

from typing import List
from typing import Union
from typing import Iterable


log = logging.get("kgc.trainer")
tqdm = partial(_tqdm, ncols=80)


@helper.notnone
def resolve_device(*, device_name: str = None):
    if device_name not in ("cuda", "cpu"):
        raise ryn.RynError(f'unknown device option: "{device_name}"')

    if not torch.cuda.is_available() and device_name == "cuda":
        log.error("cuda is not available; falling back to cpu")
        device_name = "cpu"

    device = torch.device(device_name)
    log.info(f"resolved device, running on {device}")

    return device


# --------------------


@helper.notnone
def single(
    *,
    config: Config = None,
    keen_dataset: keen.Dataset = None,
    split_dataset: split.Dataset = None,
) -> data.TrainingResult:

    # TODO https://github.com/pykeen/pykeen/issues/129
    # monkey-patched pykeen to return constant batch size
    BATCH_SIZE = 200

    # preparation

    if not config.general.seed:
        # choice of range is arbitrary
        config.general.seed = np.random.randint(10 ** 5, 10 ** 7)
        log.info(f"setting seed to {config.general.seed}")

    helper.seed(config.general.seed)

    # initialization

    result_tracker = config.resolve(config.tracker)
    result_tracker.start_run()
    result_tracker.log_params(dataclasses.asdict(config))

    device = resolve_device(
        device_name=config.model.kwargs["preferred_device"]
    )

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
        evaluation_batch_size=BATCH_SIZE,
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

        losses = training_loop.train(
            **{
                **dataclasses.asdict(config.training),
                **dict(
                    stopper=stopper,
                    result_tracker=result_tracker,
                    clear_optimizer=True,
                ),
            }
        )

    except RuntimeError as exc:
        log.error(f'training error: "{exc}"')
        log.error("sweeping training loop memory up under the rug")

        # not working although documented?
        # result_tracker.wandb.alert(title='RuntimeError', text=msg)
        result_tracker.run.finish(exit_code=1)

        gc.collect()
        training_loop.optimizer.zero_grad()
        training_loop._free_graph_and_cache()

        raise exc

    training_time = data.Time(start=ts, end=datetime.now())
    result_tracker.log_metrics(
        prefix="validation",
        metrics=dict(best=stopper.best_metric, metric=stopper.metric),
        step=stopper.best_epoch,
    )

    # aggregation

    return data.TrainingResult(
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
    resume: bool = None,
) -> optuna.Study:

    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "optuna.db"

    # removed timestamp: current way of doing it in ryn
    # has seperate optuna.db for each study; might change
    # at some point...

    # timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M')
    # study_name = f'{config.model.cls}-sweep-{timestamp}'

    study_name = f"{config.model.cls.lower()}-sweep"
    log.info(f'create optuna study "{study_name}"')

    if resume:
        log.info("! resuming old study")

    # TODO use direction="maximise"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=resume,
        # TODO make option
    )

    # if there are any initial values to be set,
    # create and enqueue a custom trial
    if not resume:
        params = {
            k: v.initial
            for k, v in config.suggestions.items()
            if v.initial is not None
        }

        if params:
            log.info(
                "setting initial study params: "
                + ", ".join(f"{k}={v}" for k, v in params.items())
            )
            study.enqueue_trial(params)

    return study


@helper.notnone
def multi(
    *,
    base: Config = None,
    out: pathlib.Path = None,
    resume: bool = False,
    **kwargs,
) -> None:

    # Optuna lingo:
    #   Trial: A single call of the objective function
    #   Study: An optimization session, which is a set of trials
    #   Parameter: A variable whose value is to be optimized
    assert base.optuna, "no optuna config found"

    def objective(trial):
        log.info(f"! starting trial {trial.number}")

        # obtain optuna suggestions
        config = base.suggest(trial)
        name = f"{config.model.cls.lower()}-{trial.number}"
        path = out / f"trial-{trial.number:04d}"

        # update configuration
        config.tracker.kwargs["experiment"] = name
        # tracker = dataclasses.replace(config.tracker, experiment=name)
        # config = dataclasses.replace(config, tracker=tracker)

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
        log.info(
            f"! trial {trial.number} finished: " f"best metric = {best_metric}"
        )

        # min optimization
        result.save(path)
        return -best_metric if base.optuna.maximise else best_metric

    study = _create_study(config=base, out=out, resume=resume)

    study.optimize(
        objective,
        n_trials=base.optuna.trials,
        gc_after_trial=True,
        catch=(ryn.RynError,),
    )

    log.info("finished study")


@helper.notnone
def train(
    *,
    config: Config = None,
    split_dataset: split.Dataset = None,
    keen_dataset: keen.Dataset = None,
    **kwargs,
) -> None:

    out = ryn.ENV.KGC_DIR / split_dataset.name / f"{config.model.cls.lower()}"
    config.save(out)

    multi(
        out=out,
        base=config,
        keen_dataset=keen_dataset,
        split_dataset=split_dataset,
        **kwargs,
    )


@helper.notnone
def train_from_kwargs(
    *,
    config: str = None,
    split_dataset: str = None,
    participate: bool,
    **kwargs,
):

    log.info("running training")
    split_dataset, keen_dataset = data.load_datasets(path=split_dataset)

    print(f"\n{split_dataset}\n{keen_dataset}\n")

    path = helper.path(config)
    config = Config.load(path.parent, fname=path.name)
    config.general.dataset = split_dataset.name

    train(
        config=config,
        split_dataset=split_dataset,
        keen_dataset=keen_dataset,
        resume=participate,
        **kwargs,
    )


@helper.notnone
def resume_from_kwargs(
    *, path: str = None, split_dataset: str = None, **kwargs
):

    log.info("resuming training")
    out = helper.path(path, exists=True)

    config = Config.load(out)
    split_dataset, keen_dataset = data.load_datasets(path=split_dataset)

    assert split_dataset.name == config.general.dataset
    multi(
        out=out,
        base=config,
        resume=True,
        keen_dataset=keen_dataset,
        split_dataset=split_dataset,
        **kwargs,
    )

    log.info("done")


# --------------------


def evaluate(
    model: torch.nn.Module = None,
    config: Config = None,
    mapped_triples=None,
    tqdm_kwargs=None,
    **kwargs,
):

    tqdm_kwargs = tqdm_kwargs or {}

    evaluator = config.resolve(
        config.evaluator,
    )

    ts = datetime.now()
    model = model.to("cuda")  # TODO (inter-machine problem)
    metrics = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        tqdm_kwargs=tqdm_kwargs,
        **kwargs,
    )

    evaluation_time = data.Time(start=ts, end=datetime.now())

    evaluation_result = data.EvaluationResult(
        model=config.model.cls,
        created=datetime.now(),
        evaluation_time=evaluation_time,
        git_hash=helper.git_hash(),
        metrics=dataclasses.asdict(metrics),
    )

    log.info(f"evaluation took: {evaluation_time.took}")
    return evaluation_result


@helper.notnone
def _get_mapped_triples(keen_dataset: keen.Dataset = None, mode: str = None):

    if mode == "testing":
        selection = (keen_dataset.testing,)

    elif mode == "validation":
        selection = (keen_dataset.validation,)

    elif mode == "all":
        selection = (
            keen_dataset.training,
            keen_dataset.validation,
            keen_dataset.testing,
        )
    else:
        raise ryn.RynError(f"unknown mode: '{mode}'")

    ref = keen_dataset.training.mapped_triples
    mapped_triples = torch.zeros((0, ref.shape[1]), dtype=ref.dtype)
    for kind in filter(None, selection):
        log.info(f"adding {len(kind.mapped_triples)} triples")
        mapped_triples = torch.vstack((mapped_triples, kind.mapped_triples))

    return mapped_triples


@helper.notnone
def evaluate_glob(
    *,
    glob: Iterable[pathlib.Path] = None,
    split_dataset: split.Dataset = None,
    keen_dataset: keen.Dataset = None,
    mode: str = None,
):

    glob = list(glob)
    log.info(f"probing {len(glob)} directories")

    results = []
    for path in tqdm(glob):
        try:
            train_result = data.TrainingResult.load(path)
            assert train_result.config.general.dataset == split_dataset.name

        except (FileNotFoundError, NotADirectoryError) as exc:
            log.info(f"skipping {path.name}: {exc}")
            continue

        try:
            eval_result = data.EvaluationResult.load(path, prefix=mode)

        except FileNotFoundError:

            mapped_triples = _get_mapped_triples(
                keen_dataset=keen_dataset, mode=mode
            )
            log.info(f"evaluting {len(mapped_triples)} triples")

            eval_result = evaluate(
                model=train_result.model,
                config=train_result.config,
                mapped_triples=mapped_triples,
                tqdm_kwargs=dict(
                    position=1,
                    ncols=80,
                    leave=False,
                ),
            )
            eval_result.save(path, prefix=mode)

        results.append((eval_result, path))

    return results


@helper.notnone
def evaluate_from_kwargs(
    *,
    results: List[str] = None,
    split_dataset: str = None,
    mode: str = None,
):

    split_dataset, keen_dataset = data.load_datasets(path=split_dataset)

    eval_results = evaluate_glob(
        glob=map(pathlib.Path, results),
        split_dataset=split_dataset,
        keen_dataset=keen_dataset,
        mode=mode,
    )

    return eval_results


@helper.notnone
def print_results(
    *, results, out: Union[str, pathlib.Path] = None, mode: str = None
):
    def _save_rget(dic, *args, default=None):
        ls = list(args)[::-1]

        while ls:
            try:
                dic = {str(k): v for k, v in dic.items()}[ls.pop()]
            except KeyError as exc:
                log.error(str(exc))
                return default

        return dic

    rows = []
    sort_key = 4  # hits@10

    headers = [
        "name",
        "model",
        "val",
        "hits@1",
        "hits@10",
        "MRR",
        "MR",
        "A-MR",
    ]
    headers[sort_key] += " *"

    for eval_result, path in results:
        train_result = data.TrainingResult.load(path, load_model=False)
        get_test = partial(_save_rget, eval_result.metrics)

        rows.append(
            [
                path.name,
                eval_result.model,
                train_result.stopper["results"][-1],
                get_test("hits_at_k", "both", "avg", "1", default=0) * 100,
                get_test("hits_at_k", "both", "avg", "10", default=0) * 100,
                get_test("mean_reciprocal_rank", "both", "avg") * 100,
                int(get_test("mean_rank", "both", "avg")),
                get_test("adjusted_mean_rank", "both", default=2),
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
