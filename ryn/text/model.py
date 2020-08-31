# -*- coding: utf-8 -*-


from ryn.common import logging

import gzip
import json
import pathlib

from datetime import datetime
from dataclasses import dataclass

import h5py

from typing import Dict
from typing import Union


log = logging.get('text.model')


@dataclass
class Data:
    """

    Tokenized text data ready to be used by a model

    This data is produced by ryn.text.encoder.transform.
    Files required for loading:

      - info.txt
      - idxs.h5

    """

    created: datetime
    git_hash: str

    model: str
    dataset: str
    database: str

    sentences: int
    tokens: int

    no_context: Dict[int, str]

    h5fd: h5py.File

    @property
    def name(self) -> str:
        return (
            f'{self.dataset}/{self.database}/'
            f'{self.model}.{self.sentences}.{self.tokens}')

    def close(self):
        log.info(f'closing model.Data for {self.name}')
        self.h5fd.close()

    @classmethod
    def load(K, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)

        with (path / 'info.json').open(mode='r') as fd:
            info = json.load(fd)

        with gzip.open(str(path / 'nocontext.txt.gz'), mode='r') as fd:
            fd.readline()  # skip head comment

            no_context = {
                int(k): v for k, v in (
                    line.split(' ', maxsplit=1) for line in (
                        fd.read().decode().strip().split('\n')))}

        h5fd = h5py.File(str(path / 'idxs.h5'), mode='r')

        data = Data(
            created=datetime.fromisoformat(info['created']),
            git_hash=info['git_hash'],
            model=info['model'],
            dataset=info['dataset'],
            database=info['database'],
            sentences=info['sentences'],
            tokens=info['tokens'],
            no_context=no_context,
            h5fd=h5fd, )

        log.info('loaded {data.name}')

        return data


@dataclass
class Embedded:
    """

    Encoded tokens

    """
    data: Data
