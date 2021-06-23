# -*- coding: utf-8 -*-

import irt
import irtm

from irtm.text import mapper
from irtm.text.config import Config
from irtm.common import helper

import pytorch_lightning as pl

import os
import logging
import pathlib
import dataclasses
from datetime import datetime

from typing import List
from typing import Optional

log = logging.getLogger(__name__)


def load_from_config(config: Config):
    ids = irt.Dataset(path=config.dataset)
    kow = irt.KeenOpenWorld(dataset=ids)

    datamodule = irt.TorchModule(
        kow=kow,
        model_name=config.text_encoder,
        dataloader_train_args=config.dataloader_train_args,
        dataloader_valid_args=config.dataloader_valid_args,
        dataloader_test_args=config.dataloader_test_args,
    )

    upstream = mapper.UpstreamModels.load(config=config, dataset=ids)
    irtmc = mapper.Components.create(config=config, upstream=upstream)

    return datamodule, irtmc


def _add_loghandler(config: Config, name: str):
    # add an additional text logger
    log.info(f"adding an additional log handler: {name}.log")

    loghandler = logging.FileHandler(
        str(pathlib.Path(config.out) / f"{name}.log"),
        mode="w",
    )

    loghandler.setLevel(log.getEffectiveLevel())
    loghandler.setFormatter(log.root.handlers[0].formatter)
    logging.getLogger("irtm").addHandler(loghandler)


def _init_logger(
    debug: bool,
    timestamp: str,
    config: Config,
    kgc_model_name: str,
    text_encoder_name: str,
    text_dataset_name: str,
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
    datamodule: irt.TorchModule,
    out: pathlib.Path,
    debug: bool,
):

    try:
        log.info("✝ rise, if you would")
        trainer.fit(model, datamodule=datamodule)

    except Exception as exc:
        log.error(f"{exc}")
        if not debug:
            with (out / "exception.txt").open(mode="w") as fd:
                fd.write(f"Exception: {datetime.now()}\n\n")
                fd.write(str(exc))

        raise exc

    if not debug:
        with (out / "profiler_summary.txt").open(mode="w") as fd:
            fd.write(trainer.profiler.summary())


def train(*, config: Config, debug: bool = False):
    irtmod, irtmc = load_from_config(config=config)
    pl.seed_everything(irtmod.kow.dataset.config.seed)

    out_fmt = config.out or (
        f"{irtm.ENV.TEXT_DIR}/mapper/{{text_dataset}}/{{text_model}}/{{kgc_model}}"
    )

    out = helper.path(
        out_fmt.format(
            **dict(
                text_dataset=irtmod.kow.dataset.name,
                text_model=irtmod.model_name,
                kgc_model=irtmc.kgc_model_name,
            )
        )
    )

    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    out = helper.path(out / timestamp, create=(not debug))
    config = dataclasses.replace(config, out=out)

    if not debug:
        log.info(f"writing data to {out}")
        out.mkdir(parents=True, exist_ok=True)
        _add_loghandler(config=config, name="training")

    map_model = mapper.Mapper(
        irtmc=irtmc,
        irtmod=irtmod,
        freeze_text_encoder=config.freeze_text_encoder,
    )

    # --

    logger = _init_logger(
        debug=debug,
        config=config,
        timestamp=timestamp,
        kgc_model_name=irtmc.kgc_model_name,
        text_encoder_name=irtmod.model_name,
        text_dataset_name=irtmod.kow.dataset.name,
        resume=False,
    )

    trainer = _init_trainer(
        config=config,
        logger=logger,
        debug=debug,
    )

    if not debug:
        dataclasses.replace(config, out=str(out)).save(out / "config.yml")

    _fit(
        trainer=trainer,
        model=map_model,
        datamodule=irtmod,
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
    irtmod, irtmc = load_from_config(config=config)

    checkpoint = helper.path(
        checkpoint,
        exists=True,
        message="loading model checkpoint {path_abbrv}",
    )

    map_model = mapper.Mapper.load_from_checkpoint(
        str(checkpoint),
        irtmc=irtmc,
        irtmod=irtmod,
        freeze_text_encoder=config.freeze_text_encoder,
    )

    timestamp = out.name
    pl.seed_everything(irtmod.split.cfg.seed)

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
        text_dataset_name=irtmod.text.name,
        resume=True,
    )

    _add_loghandler(config=config, name="resume")

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
        datamodule=irtmod,
        out=out,
        debug=debug,
    )

    log.info("resumed training finished")
