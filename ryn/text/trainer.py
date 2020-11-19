# -*- coding: utf-8 -*-

import ryn
from ryn.text import data
from ryn.text import mapper
from ryn.text.config import Config
from ryn.common import helper
from ryn.common import logging

import pytorch_lightning as pl
import horovod.torch as hvd

import pathlib
import dataclasses
from datetime import datetime

from typing import Optional

log = logging.get('text.trainer')


@helper.notnone
def _load_from_config(*, config: Config = None):
    upstream_models = data.Models.load(config=config)

    datasets = data.Datasets.load(config=config, models=upstream_models)
    rync = mapper.Components.create(config=config, models=upstream_models)

    return datasets, rync


@helper.notnone
def _init_logger(
        debug: bool = None,
        timestamp: str = None,
        config: Config = None,
        kgc_model_name: str = None,
        text_encoder_name: str = None,
        text_dataset_name: str = None,
):

    # --

    logger = None
    name = f'{text_encoder_name}.{kgc_model_name}.{timestamp}'

    if debug:
        log.info('debug mode; not using any logger')
        return None

    if config.wandb_args:
        config = dataclasses.replace(config, wandb_args={
            **dict(
                name=name,
                save_dir=str(config.out),
            ),
            **config.wandb_args, })

        log.info('initializating logger: '
                 f'{config.wandb_args["project"]}/{config.wandb_args["name"]}')

        logger = pl.loggers.wandb.WandbLogger(**config.wandb_args)
        logger.experiment.config.update({
            'kgc_model': kgc_model_name,
            'text_dataset': text_dataset_name,
            'text_encoder': text_encoder_name,
            'mapper_config': dataclasses.asdict(config),
        })

    else:
        log.info('! no wandb configuration found; falling back to csv')
        logger = pl.loggers.csv_logs.CSVLogger(config.out / 'csv', name=name)

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
        log.info(f'registering checkpoint callback: {config.checkpoint_args}')
        callbacks.append(pl.callbacks.ModelCheckpoint(
            **config.checkpoint_args))

    trainer_args = dict(
        callbacks=callbacks,
        deterministic=True,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    if not debug:
        trainer_args.update(
            profiler='simple',
            logger=logger,
            # trained model directory
            weights_save_path=config.out / 'weights',
            # checkpoint directory
            default_root_dir=config.out / 'checkpoints',
        )

    log.info('initializing trainer')
    return pl.Trainer(
        **{
            **config.trainer_args,
            **trainer_args,
        }
    )


@helper.notnone
def _fit(
        *,
        trainer: pl.Trainer = None,
        model: pl.LightningModule = None,
        datasets: data.Datasets = None,
        out: pathlib.Path = None,
        debug: bool = None
):
    log.info('pape satan, pape satan aleppe')

    try:
        trainer.fit(model, datasets.text_train, datasets.text_valid)

    except Exception as exc:
        log.error(f'{exc}')
        with (out / 'exception.txt').open(mode='w') as fd:
            fd.write(f'Exception: {datetime.now()}\n\n')
            fd.write(str(exc))

        raise exc

    if not debug:
        with (out / 'profiler_summary.txt').open(mode='w') as fd:
            fd.write(trainer.profiler.summary())


@helper.notnone
def train(*, config: Config = None, debug: bool = False):
    log.info('lasciate ogni speranza o voi che entrate')

    datasets, rync = _load_from_config(config=config)

    map_model = mapper.Mapper(
        datasets=datasets,
        rync=rync,
        freeze_text_encoder=config.freeze_text_encoder,
    )

    pl.seed_everything(datasets.split.cfg.seed)

    assert config.text_encoder == datasets.text.model
    assert datasets.text.ratio == config.valid_split, 'old cache file?'

    # --

    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

    out_dir = pathlib.Path((
        ryn.ENV.TEXT_DIR / 'mapper' /
        datasets.text.dataset /
        datasets.text.database /
        datasets.text.model /
        rync.kgc_model_name
    ))

    out = out_dir / timestamp

    if not debug:
        config = dataclasses.replace(config, out=helper.path(
            out, create=True,
            message='writing model to {path_abbrv}'))

    logger = _init_logger(
        debug=debug,
        config=config,
        timestamp=timestamp,
        kgc_model_name=rync.kgc_model_name,
        text_encoder_name=datasets.text_encoder,
        text_dataset_name=datasets.text.name
    )

    trainer = _init_trainer(
        config=config,
        logger=logger,
        debug=debug,
    )

    # hvd is initialized now

    if not debug and hvd.local_rank() == 0:
        config.save(out / 'config.json')

    _fit(
        trainer=trainer,
        model=map_model,
        datasets=datasets,
        out=out,
        debug=debug,
    )

    log.info('training finished')


@helper.notnone
def train_from_kwargs(
        debug: bool = False,
        offline: bool = False,
        kgc_model: str = None,
        text_dataset: str = None,
        split_dataset: str = None,
):

    if debug:
        log.warning('phony debug run!')

    if offline:
        log.warning('offline run!')

    # bert-large-cased: hidden size 1024
    # bert-base-cased: hidden size 768

    # --------------------

    # 24G 30 ctxs batch sizes:
    # batch_sizes = dict(
    #     training=60,
    #     validation=60,
    #     inductive=30,
    # )

    # batch_sizes = dict(
    #     training=90,
    #     validation=80,
    #     inductive=50,
    # )

    # --------------------

    # 11G 30 ctxs batch size:
    batch_sizes = dict(
        training=25,
        validation=25,
        inductive=15,
    )

    # 11G 2 ctxs batch size:
    # batch_sizes = dict(
    #     training=40,
    #     validation=40,
    #     inductive=25,
    # )

    # --------------------

    config = Config(

        # this is annoying to be declared explicitly
        # but simplifies a lot down the line
        text_encoder='bert-base-cased',

        freeze_text_encoder=False,
        valid_split=0.7,

        wandb_args=dict(
            project='ryn-text',
            log_model=False,
            offline=offline,
        ),

        trainer_args=dict(
            gpus=1,
            max_epochs=200,
            fast_dev_run=debug,
            # check_val_every_n_epoch=10,
            distributed_backend='horovod',
            accumulate_grad_batches=10,
            gradient_clip_val=1,
        ),

        checkpoint_args=dict(
            monitor='valid_loss_step',
            save_top_k=10,
        ),

        dataloader_train_args=dict(
            num_workers=0,
            batch_size=batch_sizes['training'],
            shuffle=True,
        ),

        dataloader_valid_args=dict(
            num_workers=0,
            batch_size=batch_sizes['validation'],
        ),

        dataloader_inductive_args=dict(
            num_workers=0,
            batch_size=batch_sizes['inductive'],
        ),

        # ryn upstream
        kgc_model=kgc_model,
        text_dataset=text_dataset,
        split_dataset=split_dataset,

        # pytorch
        optimizer='adam',
        optimizer_args=dict(lr=0.00001),

        # ryn models
        aggregator='cls 1',

        # projector='affine 1',
        # projector_args=dict(
        #     input_dims=768,
        #     output_dims=450,
        # ),

        projector='mlp 1',
        projector_args=dict(
            input_dims=768,
            hidden_dims=600,
            output_dims=450),

        comparator='euclidean 1',
    )

    train(config=config, debug=debug)


@helper.notnone
def resume_from_kwargs(
        path: str = None,
        checkpoint: str = None,
        debug: bool = None,
        offline: bool = None,
):

    out = helper.path(path, exists=True)
    config = Config.load(out / 'config.json')

    config.out = out
    config.wandb_args.update(dict(
        offline=offline,
    ))

    datasets, rync = _load_from_config(config=config)

    helper.path(
        checkpoint, exists=True,
        message='loading model checkpoint {path_abbrv}')

    map_model = mapper.Mapper.load_from_checkpoint(
        checkpoint,
        datasets=datasets,
        rync=rync,
        freeze_text_encoder=config.freeze_text_encoder,
    )

    timestamp = out.name

    logger = _init_logger(
        debug=debug,
        timestamp=timestamp,
        config=config,
        kgc_model_name=rync.kgc_model_name,
        text_encoder_name=datasets.text_encoder,
        text_dataset_name=datasets.text.name,
    )

    trainer = _init_trainer(
        config=config,
        logger=logger,
        debug=debug,
        resume_from_checkpoint=checkpoint,
    )

    _fit(
        trainer=trainer,
        model=map_model,
        datasets=datasets,
        out=out,
        debug=debug,
    )

    log.info('resumed training finished')
