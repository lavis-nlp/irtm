# -*- coding: utf-8 -*-

import irtm
from irtm.text import data
from irtm.text import mapper
from irtm.text.config import Config
from irtm.common import helper
from irtm.common import logging

import pytorch_lightning as pl
import horovod.torch as hvd

import os
import pathlib
import dataclasses
from datetime import datetime

from typing import List
from typing import Optional

log = logging.get("text.trainer")


def load_from_config(config: Config):
    upstream_models = data.Models.load(config=config)

    datamodule = data.DataModule(
        config=config,
        keen_dataset=upstream_models.kgc_model.keen_dataset,
        split_dataset=upstream_models.kgc_model.split_dataset,
    )

    irtmc = mapper.Components.create(config=config, models=upstream_models)

    return datamodule, irtmc


def _init_logger(
    debug: bool,
    timestamp: str,
    config: Config,
    kgc_model_name: str,
    text_encoder_name: str,
    text_dataset_name: str,
    text_dataset_identifier: str,
    resume: bool,
):

    # --

    logger = None
    name = f"{timestamp}"

    if debug:
        log.info("debug mode; not using any logger")
        return None

    if config.wandb_args:
        config = dataclasses.replace(
            config,
            wandb_args={
                **dict(
                    name=name,
                    save_dir=str(config.out),
                ),
                **config.wandb_args,
            },
        )

        log.info(
            "initializating logger: "
            f'{config.wandb_args["project"]}/{config.wandb_args["name"]}'
        )

        logger = pl.loggers.wandb.WandbLogger(**config.wandb_args)
        logger.experiment.config.update(
            {
                "kgc_model": kgc_model_name,
                "text_dataset": text_dataset_name,
                "text_encoder": text_encoder_name,
                "mapper_config": dataclasses.asdict(config),
            },
            allow_val_change=resume,
        )

    else:
        log.info("! no wandb configuration found; falling back to csv")
        logger = pl.loggers.csv_logs.CSVLogger(config.out / "csv", name=name)

    assert logger is not None
    return logger


def _init_trainer(
    config: Config = None,
    logger: Optional = None,
    debug: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> pl.Trainer:

    callbacks = []

    if not debug and config.checkpoint_args:
        log.info(f"registering checkpoint callback: {config.checkpoint_args}")
        callbacks.append(pl.callbacks.ModelCheckpoint(**config.checkpoint_args))

    callbacks.append(
        pl.callbacks.LearningRateMonitor(
            logging_interval="step",
        )
    )

    if config.early_stopping:
        log.info(f"adding early stopping: '{config.early_stopping_args}'")
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(**config.early_stopping_args)
        )

    trainer_args = dict(
        callbacks=callbacks,
        deterministic=True,
        resume_from_checkpoint=resume_from_checkpoint,
        fast_dev_run=debug,
    )

    if not debug:
        trainer_args.update(
            profiler="simple",
            logger=logger,
            # trained model directory
            weights_save_path=config.out / "weights",
            # checkpoint directory
            default_root_dir=config.out / "checkpoints",
        )

    log.info(f"initializing trainer with {config.trainer_args}")
    return pl.Trainer(
        **{
            **config.trainer_args,
            **trainer_args,
        }
    )


def _fit(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: data.DataModule,
    out: pathlib.Path,
    debug: bool,
):
    log.info("pape satan, pape satan aleppe")

    try:
        trainer.fit(model, datamodule=datamodule)

    except Exception as exc:
        log.error(f"{exc}")
        with (out / "exception.txt").open(mode="w") as fd:
            fd.write(f"Exception: {datetime.now()}\n\n")
            fd.write(str(exc))

        raise exc

    if not debug:
        with (out / "profiler_summary.txt").open(mode="w") as fd:
            fd.write(trainer.profiler.summary())


def train(*, config: Config, debug: bool = False):
    log.info("lasciate ogni speranza o voi che entrate")

    datamodule, irtmc = load_from_config(config=config)

    map_model = mapper.Mapper(
        irtmc=irtmc,
        data=datamodule,
        freeze_text_encoder=config.freeze_text_encoder,
    )

    pl.seed_everything(datamodule.split.cfg.seed)

    # --

    out_fmt = config.out or (
        f"{irtm.ENV.TEXT_DIR}/mapper/"
        "{text_dataset}/{text_database}/"
        "{text_model}/{kgc_model}"
    )

    out = helper.path(
        out_fmt.format(
            **dict(
                text_dataset=datamodule.text.dataset,
                text_database=datamodule.text.database,
                text_model=datamodule.text.model,
                kgc_model=irtmc.kgc_model_name,
            )
        )
    )

    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    out = helper.path(
        out / timestamp, create=True, message="writing model to {path_abbrv}"
    )

    config = dataclasses.replace(config, out=out)

    # --

    logger = _init_logger(
        debug=debug,
        config=config,
        timestamp=timestamp,
        kgc_model_name=irtmc.kgc_model_name,
        text_encoder_name=config.text_encoder,
        text_dataset_name=datamodule.text.name,
        text_dataset_identifier=datamodule.text.identifier,
        resume=False,
    )

    trainer = _init_trainer(
        config=config,
        logger=logger,
        debug=debug,
    )

    # hvd is initialized now

    if not debug and hvd.local_rank() == 0:
        dataclasses.replace(config, out=str(out)).save(out / "config.yml")

    if not hvd.local_rank() == 0:
        logger = None

    _fit(
        trainer=trainer,
        model=map_model,
        datamodule=datamodule,
        out=out,
        debug=debug,
    )

    log.info("training finished")


def train_from_kwargs(
    config: List[str],
    debug: bool = False,
    **kwargs,
):

    if debug:
        log.warning("debug run")

    config = Config.create(configs=config, **kwargs)
    train(config=config, debug=debug)


def resume_from_kwargs(
    path: str,
    checkpoint: str,
    debug: bool,
    config: List[str],
    **kwargs,
):

    out = helper.path(path, exists=True)
    config_file = out / "config.yml"
    config = Config.create(configs=[config_file] + list(config), **kwargs)

    config.out = out
    datamodule, irtmc = load_from_config(config=config)

    checkpoint = helper.path(
        checkpoint,
        exists=True,
        message="loading model checkpoint {path_abbrv}",
    )

    map_model = mapper.Mapper.load_from_checkpoint(
        str(checkpoint),
        data=datamodule,
        irtmc=irtmc,
        freeze_text_encoder=config.freeze_text_encoder,
    )

    timestamp = out.name
    pl.seed_everything(datamodule.split.cfg.seed)

    # it is not possible to set resume=... for wandb.init
    # with pytorch lightning - so we need to fumble around
    # with os.environ...
    # (see pytorch_lightning/loggers/wandb.py:127)
    run_id = checkpoint.parent.parent.name
    os.environ["WANDB_RUN_ID"] = run_id
    log.info(f"! resuming from run id: {run_id}")

    logger = _init_logger(
        debug=debug,
        timestamp=timestamp,
        config=config,
        kgc_model_name=irtmc.kgc_model_name,
        text_encoder_name=config.text_encoder,
        text_dataset_name=datamodule.text.name,
        text_dataset_identifier=datamodule.text.identifier,
        resume=True,
    )

    trainer = _init_trainer(
        config=config,
        logger=logger,
        debug=debug,
        resume_from_checkpoint=str(checkpoint),
    )

    helper.path_rotate(config_file)
    config = dataclasses.replace(config, out=str(out))
    config.save(config_file)

    _fit(
        trainer=trainer,
        model=map_model,
        datamodule=datamodule,
        out=out,
        debug=debug,
    )

    log.info("resumed training finished")
