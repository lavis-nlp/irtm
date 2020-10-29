# -*- coding: utf-8 -*-

import ryn
from ryn.kgc import keen
from ryn.kgc import pipeline
from ryn.kgc.config import Config
from ryn.graphs import split
from ryn.common import helper
from ryn.common import logging

from datetime import datetime


log = logging.get('kgc.trainer')


@helper.notnone
def train(
        *,
        base: Config = None,
        split_dataset: split.Dataset = None,
        keen_dataset: keen.Dataset = None,
        offline: bool = False,
) -> None:

    time = str(datetime.now()).replace(' ', '_')
    out = ryn.ENV.KGC_DIR / split_dataset.name / f'{base.model.cls}-{time}'
    base.save(out)

    pipeline.multi(
        out=out,
        base=base,
        split_dataset=split_dataset,
        keen_dataset=keen_dataset)


def train_from_cli(
        config: str = None,
        split_dataset: str = None,
        offline: bool = False):
    log.info('running training from cli')

    if offline:
        log.warning('offline run!')

    split_dataset = split.Dataset.load(path=split_dataset)
    keen_dataset = keen.Dataset.create(
        name=split_dataset.name,
        path=split_dataset.path,
        split_dataset=split_dataset)

    print(f'\n{split_dataset}\n{keen_dataset}\n')

    base = Config.load(config)
    base.general.dataset = split_dataset.name

    train(
        base=base,
        split_dataset=split_dataset,
        keen_dataset=keen_dataset,
        offline=offline)
