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
# from pytorch_lightning.loggers.wandb import WandbLogger as Logger

import gc
import argparse

from itertools import chain
from itertools import repeat
from functools import partial
from datetime import datetime

from typing import List
from typing import Tuple

log = logging.get('text.trainer')


class Dataset(torch_data.Dataset):

    @property
    def token_ids(self) -> Tuple[torch.Tensor]:
        return self._token_ids

    @property
    def entity_ids(self) -> Tuple[int]:
        return self._entity_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return (
            self.token_ids[idx],
            self.entity_ids[idx], )

    def __init__(
            self, *,

            # either
            part: data.Part = None,
            # or
            entity_ids: Tuple[int] = None,
            token_ids: Tuple[torch.Tensor] = None):

        if any((
                not part and not (token_ids and entity_ids),
                part and (token_ids or entity_ids), )):
            assert False, 'provide either a part or token & entity ids'

        if part:
            flat = [
                (e, torch.Tensor(token_ids).to(dtype=torch.long))
                for e, id_lists in part.id2idxs.items()
                for token_ids in id_lists]

            self._entity_ids, self._token_ids = zip(*flat)

        else:
            self._entity_ids = entity_ids
            self._token_ids = token_ids

        assert len(self.entity_ids) == len(self.token_ids)

    def split(self, ratio: float) -> Tuple['Dataset']:
        # must not be shuffled

        n = int(len(self.entity_ids) * ratio)
        e = self.entity_ids[n]

        while self.entity_ids[n] == e:
            n += 1

        log.info(
            f'splitting dataset with param {ratio}'
            f' at {n} ({n / len(self.entity_ids) * 100:2.2f}%)')

        a = Dataset(
            entity_ids=self.entity_ids[:n],
            token_ids=self.token_ids[:n])

        b = Dataset(
            entity_ids=self.entity_ids[n:],
            token_ids=self.token_ids[n:])

        assert not (set(a.entity_ids) & set(b.entity_ids))
        return a, b


def collate_fn(batch: List[Tuple]):
    idxs, ents = zip(*batch)
    return pad_sequence(idxs, batch_first=True), ents


OPTIMIZER = {
    'adam': torch.optim.Adam,
}


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

    text_dataset = data.Dataset.load(path=config.text_dataset)

    model = mapper.Mapper.from_config(
        config=config,
        text_encoder_name=text_dataset.model)

    # # TODO make option
    model.c.text_encoder.eval()

    # to reproduce runs:
    # pl.seed_everything(model.c.kgc_model.ds.cfg.seed)
    # also pl.Trainer(deterministic=True, ...)

    assert model.c.tokenizer.base.vocab['[PAD]'] == 0

    # handling data

    ds = Dataset(part=text_dataset.cw_train)
    ds_train, ds_valid = ds.split(config.valid_split)
    # test = Dataset(part=text_dataset.ow_valid)

    DataLoader = partial(torch_data.DataLoader, collate_fn=collate_fn)
    dl_train = DataLoader(ds_train, **config.dataloader_train_args)
    dl_valid = DataLoader(ds_valid, **config.dataloader_valid_args)

    # torment the machine

    # callback = TrainerCallback(
    #     config=config,
    #     dl_train=dl_train,
    #     dl_valid=dl_valid)

    trainer = pl.Trainer(**config.trainer_args)
    trainer.fit(model, dl_train, dl_valid)

    log.info('done')


def train_from_args(args: argparse.Namespace):

    DEBUG = True
    if DEBUG:
        log.warning('phony debug run!')

    DATEFMT = '%Y.%m.%d.%H%M%S'

    # ---

    kgc_model = 'DistMult'
    text_encoder = 'bert-base-cased'
    split_dataset = 'oke.fb15k237_30061990_50'

    kgc_model_dir = f'{kgc_model}-256-2020.08.12.120540.777006'
    text_encoder_dir = f'{text_encoder}.200.768-small'

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

    train(config=mapper.MapperConfig(

        # pytorch lightning trainer
        # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api
        trainer_args=dict(
            max_epochs=2,
            gpus=1,
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
            ryn.ENV.EMBER_DIR / split_dataset / kgc_model_dir),

        text_dataset=(
            ryn.ENV.TEXT_DIR / 'data' / split_dataset / text_encoder_dir),

        # pytorch
        optimizer='adam',
        optimizer_args=dict(lr=0.001),

        # ryn models
        aggregator='max 1',
        projector='mlp 1',
        projector_args=dict(input_dims=768, hidden_dims=500, output_dims=256),
        comparator='euclidean 1',
        valid_split=0.7,
    ))
