# -*- coding: utf-8 -*-

import ryn
from ryn import app
from ryn.graphs import split

import pathlib
from dataclasses import dataclass

import pandas as pd
import streamlit as st


HEADER = """

## SPLITS

HyperFB Graph Splits

"""


@st.cache(allow_output_mutation=True)  # Datasets aren't hashable
def load_dataset(path: pathlib.Path) -> split.Dataset:
    return split.Dataset.load(path)


def ds_to_df(ds: split.Dataset):
    def _p(v):
        nonlocal ds
        return v * 100 // len(ds.g.source.triples)

    return [
        ds.cfg.seed, ds.cfg.threshold, len(ds.concepts),
        len(ds.cw_train.triples), _p(len(ds.cw_train.triples)),
        len(ds.cw_valid.triples), _p(len(ds.cw_valid.triples)),
        len(ds.ow_valid.triples), _p(len(ds.ow_valid.triples)),
        len(ds.ow_test.triples), _p(len(ds.ow_test.triples)),
        len(ds.ow_valid.owe), len(ds.ow_test.owe), ]


@dataclass
class SplitWidgets:

    seed: int
    threshold: int

    @classmethod
    def create(K, datasets):
        val_all = 'all'

        seeds, thresholds, *_ = [
            [val_all] + list(s) for s in
            map(set, zip(*(ds_to_df(ds) for ds in datasets)))
        ]

        def _init(fn, name, vals, mapper):
            val = fn(name, vals)
            return None if val == val_all else mapper(val)

        return K(
            seed=_init(st.sidebar.selectbox, 'Seed', seeds, int),
            threshold=_init(st.sidebar.selectbox, 'Threshold', thresholds, int)
        )


def splits_load_datasets():
    glob = list(pathlib.Path(ryn.ENV.SPLIT_DIR).glob('*/cfg.pkl'))
    st.write(f'loading {len(glob)} datasets...')

    latest_iteration = st.empty()
    bar = st.progress(0)

    datasets = []
    for i, path in enumerate(glob):
        datasets.append(load_dataset(path.parent))
        latest_iteration.text(f'Dataset {i+1}/{len(glob)}')
        bar.progress((i + 1) / len(glob))

    return datasets


def splits_filter_datasets(datasets, widgets):
    conds = (
        (widgets.seed, lambda ds: ds.cfg.seed),
        (widgets.threshold, lambda ds: ds.cfg.threshold),
    )

    for val, fn in conds:
        if val is not None:
            datasets = list(filter(lambda ds: fn(ds) == val, datasets))

    return datasets


def splits_show_datasets(datasets):
    st.write('### Overview')

    cols = [
        'seed', 'threshold', 'concepts',
        'tris:cw.train', '%', 'tris:cw.valid', '%',
        'tris:ow.valid', '%', 'trisles:ow.test', '%',
        'owe-ents:ow.valid', 'owe-ents:ow.test', ]

    pd_overview = pd.DataFrame(
        data=[ds_to_df(ds) for ds in datasets],
        columns=cols)

    st.dataframe(pd_overview)


def splits_show_dataset(ds: split.Dataset):
    st.write(f'### Details for {ds.cfg.seed}-{ds.cfg.threshold}')


def splits():
    datasets = splits_load_datasets()

    widgets = SplitWidgets.create(datasets)
    datasets = splits_filter_datasets(datasets, widgets)

    assert len(datasets) > 0

    if len(datasets) == 1:
        splits_show_dataset(datasets[0])
    else:
        splits_show_datasets(datasets)


@dataclass
class Context(app.Context):

    def run(self):
        splits()  # TODO refactor
