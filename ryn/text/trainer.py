# -*- coding: utf-8 -*-

import ryn
from ryn.text import data
from ryn.text import mapper
from ryn.text.config import Config
from ryn.common import helper
from ryn.common import logging

import torch.optim
import torch.utils.data as torch_data
import pytorch_lightning as pl
# import horovod.torch as hvd

import gc
import pathlib
import dataclasses
from datetime import datetime

from itertools import chain
from itertools import repeat

from typing import Optional

log = logging.get('text.trainer')


class TrainerCallback(pl.callbacks.base.Callback):

    @property
    def config(self):
        return self._config

    @property
    def dl_train(self) -> torch_data.DataLoader:
        return self._dl_train

    @property
    def dl_valid(self) -> torch_data.DataLoader:
        return self._dl_valid

    def __init__(
            self,
            *args,
            config: Config = None,
            dl_train: torch_data.DataLoader = None,
            dl_valid: torch_data.DataLoader = None,
            **kwargs):

        super().__init__(*args, **kwargs)
        self._config = config
        self._dl_train = dl_train
        self._dl_valid = dl_valid

    def on_sanity_check_start(self, trainer, mapper):
        log.info('probing for functioning configuration')

        max_seq = []
        for seq, _ in chain(self.dl_train.dataset, self.dl_valid.dataset):
            if len(max_seq) < len(seq):
                max_seq = seq

        log.info(f'determined max sequence length: {len(max_seq)}')

        for batch_size in set((
                self.config.dataloader_train_args['batch_size'],
                self.config.dataloader_valid_args['batch_size'], )):

            log.info(f'testing {batch_size=}')
            sentences = max_seq.repeat(batch_size, 1).to(device=mapper.device)

            mapper(
                sentences=sentences,
                entities=repeat(0, batch_size))

        log.info('clean up after probing')

        for p in mapper.parameters():
            if p.grad is not None:
                del p.grad

        torch.cuda.empty_cache()
        gc.collect()


@helper.notnone
def _init_logger(
        debug: bool = None,
        timestamp: str = None,
        config: Config = None,
        kgc_model_name: str = None,
        text_encoder_name: str = None,
        text_dataset_name: str = None,
):

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
) -> pl.Trainer:

    callbacks = []

    if not debug and config.checkpoint_args:
        log.info(f'registering checkpoint callback: {config.checkpoint_args}')
        callbacks.append(pl.callbacks.ModelCheckpoint(
            **config.checkpoint_args))

    trainer_args = dict(
        callbacks=callbacks,
        deterministic=True,
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
def train(*, config: Config = None, debug: bool = False):
    log.info('lasciate ogni speranza o voi che entrate')

    upstream_models = data.Models.load(config=config)
    datasets = data.Datasets.load(config=config, models=upstream_models)

    map_model = mapper.Mapper.create(
        config=config,
        datasets=datasets,
        models=upstream_models,
    )

    pl.seed_everything(datasets.split.cfg.seed)

    assert config.text_encoder == datasets.text.model
    assert datasets.text.ratio == config.valid_split, 'old cache file?'

    # --

    timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

    out = pathlib.Path((
        ryn.ENV.TEXT_DIR / 'mapper' /
        datasets.text.dataset /
        datasets.text.database /
        datasets.text.model /
        upstream_models.kgc_model_name /
        timestamp
    ))

    if not debug:
        config = dataclasses.replace(config, out=helper.path(
            out, create=True,
            message='writing model to {path_abbrv}'))

    log.error('LOGGING DISABLED')
    logger = _init_logger(
        debug=True,  # debug,
        config=config,
        timestamp=timestamp,
        kgc_model_name=upstream_models.kgc_model_name,
        text_encoder_name=datasets.text_encoder,
        text_dataset_name=datasets.text.name
    )

    trainer = _init_trainer(
        config=config,
        logger=logger,
        debug=debug,
    )

    # hvd is initialized now

    if not debug:
        config.save(out)

    # if not debug and hvd.local_rank() == 0:
    #     config.save(out)

    log.info('pape satan, pape satan aleppe')
    trainer.fit(map_model, datasets.text_train, datasets.text_valid)

    if not debug:
        with (out / 'profiler_summary.txt').open(mode='w') as fd:
            fd.write(trainer.profiler.summary())

    log.info('training finished')


@helper.notnone
def train_from_cli(
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

    # 24G:
    #  - train batch size: 60
    #  - valid batch size: 45

    # 11G:
    #  - train batch size: 25
    #  - valid batch size; 15

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
            max_epochs=2,
            fast_dev_run=debug,
            # auto_lr_find=True,

            # horovod
            # gpus=1,
            # distributed_backend='horovod',
            # accumulate_grad_batches=10,
        ),

        checkpoint_args=dict(
            # monitor='valid_loss',
            save_top_k=-1,  # save all
            period=250,
        ),

        dataloader_train_args=dict(
            num_workers=0,
            batch_size=25,
            shuffle=True,
        ),

        dataloader_valid_args=dict(
            num_workers=0,
            batch_size=15,
        ),

        dataloader_inductive_args=dict(
            num_workers=0,
            batch_size=15,
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

        projector='affine 1',
        projector_args=dict(
            input_dims=768,
            output_dims=450,
        ),

        # projector='mlp 1',
        # projector_args=dict(
        #     input_dims=768,
        #     hidden_dims=500,
        #     output_dims=450),

        comparator='euclidean 1',
    )

    train(config=config, debug=debug)
