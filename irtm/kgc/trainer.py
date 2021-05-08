# -*- coding: utf-8 -*-

import irt

import irtm
from irtm.kgc import data
from irtm.kgc.config import Config
from irtm.common import helper

import torch
import optuna
import numpy as np
from pykeen.pipeline import pipeline
from tqdm import tqdm as _tqdm
from tabulate import tabulate

import gc
import csv
import logging
import pathlib
import dataclasses
from functools import partial
from datetime import datetime

from typing import List
from typing import Union
from typing import Iterable
from typing import Optional


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


def resolve_device(device_name: str = None):
    if device_name not in ("cuda", "cpu"):
        raise irtm.IRTMError(f'unknown device option: "{device_name}"')

    if not torch.cuda.is_available() and device_name == "cuda":
        log.error("cuda is not available; falling back to cpu")
        device_name = "cpu"

    device = torch.device(device_name)
    log.info(f"resolved device, running on {device}")

    return device


# --------------------


def single(
    out: Union[str, pathlib.Path],
    config: Config,
    kcw: irt.KeenClosedWorld,
) -> data.TrainingResult:
    """

    A single training run

    Parameters
    ----------

    config : Config
      Configuration options

    kcw : irt.KeenClosedworld
      IRT encapsulated for closed-world KGC

    Returns
    -------

    A training result object which encapsulates
    the pykeen result tracker object

    """
    out = helper.path(out, create=True)

    # preparation

    if not config.general.seed:
        # choice of range is arbitrary
        config.general.seed = np.random.randint(10 ** 5, 10 ** 7)
        log.info(f"setting seed to {config.general.seed}")

    helper.seed(config.general.seed)

    kwargs = {}
    if config.sampler:
        kwargs["negative_sampler"] = config.sampler.cls
        kwargs["negative_sampler_kwargs"] = config.sampler.kwargs

    ts = datetime.now()

    pykeen_result = pipeline(
        training=kcw.training,
        validation=kcw.validation,
        testing=kcw.validation,
        use_testing_data=False,
        # copied from config
        model=config.model.cls,
        model_kwargs=config.model.kwargs,
        loss=config.loss.cls,
        loss_kwargs=config.loss.kwargs,
        regularizer=config.regularizer.cls,
        regularizer_kwargs=config.regularizer.kwargs,
        optimizer=config.optimizer.cls,
        optimizer_kwargs=config.optimizer.kwargs,
        training_loop=config.training_loop.cls,
        training_kwargs=dataclasses.asdict(config.training),
        stopper=config.stopper.cls,
        stopper_kwargs=config.stopper.kwargs,
        evaluator=config.evaluator.cls,
        evaluator_kwargs=config.evaluator.kwargs,
        result_tracker=config.tracker.cls,
        result_tracker_kwargs=config.tracker.kwargs,
        # hard-coded
        clear_optimizer=True,
        # automatic_memory_optimization=True,
        random_seed=config.general.seed,
        **kwargs,
    )

    # configure training result

    training_time = data.Time(start=ts, end=datetime.now())

    wandb_run = pykeen_result.stopper.result_tracker.run
    wandb = dict(
        id=wandb_run.id,
        dir=wandb_run.dir,
        path=wandb_run.path,
        name=wandb_run.name,
        offline=True,
    )

    if not wandb_run.offline:
        wandb.update(dict(url=wandb_run.url, offline=False))

    training_result = data.TrainingResult(
        created=datetime.now(),
        git_hash=helper.git_hash(),
        config=config,
        model=pykeen_result.model,
        # metrics
        training_time=training_time,
        results=pykeen_result.metric_results.to_dict(),
        losses=pykeen_result.losses,
        best_metric=pykeen_result.stopper.best_metric,
        wandb=wandb,
    )

    pykeen_result.save_to_directory(out / "pykeen")
    training_result.save(out)

    return training_result


def _create_study(
    config: Config,
    out: pathlib.Path,
    resume: bool,
) -> optuna.Study:
    """ """

    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "optuna.db"

    # removed timestamp: current way of doing it in irtm
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
            k: v.initial for k, v in config.suggestions.items() if v.initial is not None
        }

        if params:
            log.info(
                "setting initial study params: "
                + ", ".join(f"{k}={v}" for k, v in params.items())
            )
            study.enqueue_trial(params)

    return study


def multi(
    base: Config,
    out: pathlib.Path,
    resume: bool,
    **kwargs,
) -> None:
    """

    KGC training with HPO

    Conduct multiple training runs in a hyperparameter study.  Use
    config.Suggestion objects to define parameter ranges. To
    participate in an already running study (i.e. parallel training)
    or to resume an older study, set resume to True.

    Parameters
    ----------

    base : Config
      Configuration with Suggestions

    out : Union[str, pathlib.Path]
      Folder to write models and the hpo database

    resume : bool
      If False, create a new study, otherwise participate

    kwargs
      see single()

    """

    # Optuna lingo:
    #   Trial: A single call of the objective function
    #   Study: An optimization session, which is a set of trials
    #   Parameter: A variable whose value is to be optimized
    assert base.optuna, "no optuna config found"

    def objective(trial):
        log.info(f"! starting trial {trial.number}")

        # obtain optuna suggestions
        config = base.suggest(trial)

        name = (
            f"{config.general.dataset} "
            f"{config.model.cls.lower()}"
            f"-{trial.number}"
        )

        path = out / f"trial-{trial.number:04d}"

        # update configuration
        config.tracker.kwargs["experiment"] = name
        # tracker = dataclasses.replace(config.tracker, experiment=name)
        # config = dataclasses.replace(config, tracker=tracker)

        def _run(attempt: int = 1):
            # run training
            try:
                log.info(f"running attempt {attempt}")
                return single(out=out, config=config, **kwargs)

            except RuntimeError as exc:
                msg = f'objective: got runtime error "{exc}"'
                log.error(msg)

                if attempt > 3:
                    log.error("aborting attempts, something is wrong.")
                    # post mortem (TODO last model checkpoint)
                    config.save(path)
                    raise irtm.IRTMError(msg)

                log.info("releasing memory manually")
                gc.collect()
                torch.cuda.empty_cache()

                return _run(attempt=attempt + 1)

        result = _run()
        best_metric = result.best_metric
        log.info(f"! trial {trial.number} finished: " f"best metric = {best_metric}")

        # min optimization
        return -best_metric if base.optuna.maximise else best_metric

    study = _create_study(config=base, out=out, resume=resume)

    study.optimize(
        objective,
        n_trials=base.optuna.trials,
        gc_after_trial=True,
        # catch=(irtm.IRTMError,),
    )

    log.info("finished study")


def train(
    out: Union[str, pathlib.Path],
    config: Config,
    kcw: irt.KeenClosedWorld,
    **kwargs,
) -> None:
    """

    Train one or many KGC models

    """
    out = helper.path(out)
    config.save(out)

    if config.optuna:
        multi(out=out, base=config, kcw=kcw, **kwargs)

    else:
        single(out=out, config=config, kcw=kcw)


def train_from_kwargs(
    config: str,
    dataset: str,
    participate: bool,
    out: Optional[str] = None,
    **kwargs,
):
    config_path = helper.path(config)
    config = Config.load(config_path.parent, fname=config_path.name)

    dataset = irt.Dataset(dataset)
    kcw = irt.KeenClosedWorld(
        dataset=dataset,
        seed=config.general.seed or dataset.split.cfg.seed,
        split=config.general.split,
    )

    log.info(str(kcw.dataset))
    log.info(str(kcw))

    # now kith
    config.general.dataset = kcw.dataset.name
    config.general.seed = kcw.seed

    out = out or irtm.ENV.KGC_DIR / kcw.dataset.name / f"{config.model.cls.lower()}"

    train(
        kcw=kcw,
        config=config,
        resume=participate,
        out=out,
        **kwargs,
    )

    log.info("finished training")


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


def _get_mapped_triples(
    kcw: irt.KeenClosedWorld,
    mode: str,
):

    if mode == "testing":
        selection = (kcw.testing,)

    elif mode == "validation":
        selection = (kcw.validation,)

    elif mode == "all":
        selection = (
            kcw.training,
            kcw.validation,
            kcw.testing,
        )
    else:
        raise irtm.IRTMError(f"unknown mode: '{mode}'")

    ref = kcw.training.mapped_triples
    mapped_triples = torch.zeros((0, ref.shape[1]), dtype=ref.dtype)
    for kind in filter(None, selection):
        log.info(f"adding {len(kind.mapped_triples)} triples")
        mapped_triples = torch.vstack((mapped_triples, kind.mapped_triples))

    return mapped_triples


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
            train_result = data.TrainingResult.load(path)
            config = Config.load(path)

            assert config.seed

            kcw = irt.KeenClosedWorld(
                dataset=dataset,
                seed=config.general.seed,
                split=config.general.split,
            )

            assert train_result.config.general.dataset == kcw.dataset.name

        except (FileNotFoundError, NotADirectoryError) as exc:
            log.info(f"skipping {path.name}: {exc}")
            continue

        try:
            eval_result = data.EvaluationResult.load(path, prefix=mode)

        except FileNotFoundError:

            mapped_triples = _get_mapped_triples(kcw=kcw, mode=mode)
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
