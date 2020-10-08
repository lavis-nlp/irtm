# -*- coding: utf-8 -*-

import ryn
from ryn.text import data
from ryn.text import mapper
from ryn.common import helper
from ryn.common import logging

import torch.optim
import torch.utils.data as torch_data
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback

import gc
import argparse
import dataclasses

from itertools import chain
from itertools import repeat
from datetime import datetime
from collections import defaultdict

from typing import List
from typing import Tuple

log = logging.get('text.trainer')


class Dataset(torch_data.Dataset):

    def __len__(self):
        return len(self._flat)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._flat[idx]


class TrainSet(Dataset):

    @helper.notnone
    def __init__(self, *, part: data.Part = None):
        super().__init__()

        self._flat = [
            (e, torch.Tensor(idxs).to(dtype=torch.long))
            for e, idx_lists in part.id2idxs.items()
            for idxs in idx_lists]

    @staticmethod
    def collate_fn(batch: List[Tuple]):
        ents, idxs = zip(*batch)
        return pad_sequence(idxs, batch_first=True), ents


class ValidationSet(Dataset):

    def __init__(self, **parts: data.Part):
        super().__init__()
        self._flat = [
            (name, e, torch.Tensor(idxs).to(dtype=torch.long))
            for name, part in parts.items()
            for e, idx_lists in part.id2idxs.items()
            for idxs in idx_lists
        ]

    @staticmethod
    def collate_fn(batch: List[Tuple]):

        idxd = defaultdict(list)
        entd = defaultdict(list)

        for name, e, idxs in batch:
            idxd[name].append(idxs)
            entd[name].append(e)

        resd = {}
        for name in idxd:
            resd[name] = (
                pad_sequence(idxd[name], batch_first=True),
                entd[name])

        return resd


class TrainerCallback(Callback):

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
            config: mapper.MapperConfig = None,
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
def train(*, config: mapper.MapperConfig = None):
    # initializing models
    log.info('initializing models')

    text_dataset = data.Dataset.load(
        path=config.text_dataset,
        ratio=config.valid_split)

    # if a previous cache file with different ratio
    # has not been deleted prior
    assert text_dataset.ratio == config.valid_split, 'old cache file?'

    model = mapper.Mapper.from_config(
        config=config,
        text_encoder_name=text_dataset.model)

    if config.freeze_text_encoder:
        log.info('freezing text encoder')
        model.c.text_encoder.eval()

    # TODO to reproduce runs:
    # pl.seed_everything(...)
    # also pl.Trainer(deterministic=True, ...)

    assert model.c.tokenizer.base.vocab['[PAD]'] == 0

    # --- handling data

    # TODO
    # # test = Dataset(part=text_dataset.ow_valid)
    dl_train = torch_data.DataLoader(
        TrainSet(part=text_dataset.train),
        collate_fn=TrainSet.collate_fn,
        **config.dataloader_train_args)

    dl_valid = torch_data.DataLoader(
        ValidationSet(
            inductive=text_dataset.inductive,
            transductive=text_dataset.transductive),
        collate_fn=ValidationSet.collate_fn,
        **config.dataloader_valid_args)

    # --- torment the machine

    # callback = TrainerCallback(
    #     config=config,
    #     dl_train=dl_train,
    #     dl_valid=dl_valid)

    trainer = pl.Trainer(**config.trainer_args)
    trainer.fit(model, dl_train, dl_valid)

    log.info('done')


def train_from_args(args: argparse.Namespace):

    DEBUG = args.debug
    if DEBUG:
        log.warning('phony debug run!')

    DATEFMT = '%Y.%m.%d.%H%M%S'

    # ---

    kgc_model = 'DistMult'
    text_encoder = 'bert-base-cased'
    split_dataset = 'oke.fb15k237_30061990_50'

    kgc_model_dir = f'{kgc_model}-256-2020.08.12.120540.777006'
    text_encoder_dir = f'{text_encoder}.200.768-unmasked'

    # ---

    now = datetime.now().strftime(DATEFMT)
    name = f'{kgc_model.lower()}.{text_encoder.lower()}.{now}'
    out = ryn.ENV.TEXT_DIR / 'mapper' / split_dataset / name

    out.mkdir(parents=True, exist_ok=True)

    # ---

    logger = pl.loggers.wandb.WandbLogger(
        name=name,
        save_dir=str(out),
        offline=DEBUG,
        project='ryn',
        log_model=False,
    )

    # ---

    # bert-large-cased: hidden size 1024
    # bert-base-cased: hidden size 768
    config = mapper.MapperConfig(

        freeze_text_encoder=True,

        # pytorch lightning trainer
        # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api
        trainer_args=dict(
            gpus=1,
            max_epochs=25,
            logger=logger,
            weights_save_path=out / 'weights',
            # auto_lr_find=True,
            fast_dev_run=DEBUG,
        ),

        # torch dataloader
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        dataloader_train_args=dict(
            num_workers=64,
            batch_size=8,
            shuffle=True,
        ),
        dataloader_valid_args=dict(
            num_workers=64,
            batch_size=8,
        ),

        # ryn upstream
        kgc_model=(
            ryn.ENV.KGC_DIR / split_dataset / kgc_model_dir),

        text_dataset=(
            ryn.ENV.TEXT_DIR / 'data' / split_dataset / text_encoder_dir),

        # pytorch
        optimizer='adam',
        optimizer_args=dict(lr=0.00001),

        # ryn models
        aggregator='max 1',
        projector='mlp 1',
        projector_args=dict(input_dims=768, hidden_dims=500, output_dims=256),
        comparator='euclidean 1',
        valid_split=0.7,
    )

    logger.experiment.config.update({
        'kgc_model': kgc_model,
        'text_encoder': text_encoder,
        'split_dataset': split_dataset,
        'mapper_config': dataclasses.asdict(config),
    })

    train(config=config)
