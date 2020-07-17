# -*- coding: utf-8 -*-

import ryn
from ryn import app
from ryn.app import helper
from ryn.graphs import split

import random
import pathlib
from dataclasses import dataclass

import pandas as pd
import streamlit as st


@st.cache(allow_output_mutation=True)  # Datasets aren't hashable
def load_dataset(path: pathlib.Path) -> split.Dataset:
    return split.Dataset.load(path)


def ds_to_row(ds: split.Dataset):
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
            map(set, zip(*(ds_to_row(ds) for ds in datasets)))
        ]

        def _init(fn, name, vals, mapper):
            val = fn(name, vals)
            return None if val == val_all else mapper(val)

        return K(
            seed=_init(st.sidebar.selectbox, 'Seed', seeds, int),
            threshold=_init(st.sidebar.selectbox, 'Threshold', thresholds, int)
        )


class Context(app.Context):

    def load_datasets(self):
        glob = list(pathlib.Path(ryn.ENV.SPLIT_DIR).glob('*/cfg.pkl'))
        st.write(f'loading {len(glob)} datasets...')

        latest_iteration = st.empty()
        bar = st.progress(0)

        self.datasets = []
        for i, path in enumerate(glob):
            self.datasets.append(load_dataset(path.parent))
            latest_iteration.text(f'Dataset {i+1}/{len(glob)}')
            bar.progress((i + 1) / len(glob))

    def filter_datasets(self):
        conds = (
            (self.widgets.seed, lambda ds: ds.cfg.seed),
            (self.widgets.threshold, lambda ds: ds.cfg.threshold),
        )

        for val, fn in conds:
            if val is not None:
                self.datasets = list(filter(
                    lambda ds: fn(ds) == val,
                    self.datasets))

    def show_datasets(self):
        st.write('### Overview')

        cols = [
            'seed', 'threshold', 'concepts',
            'tris:cw.train', '%', 'tris:cw.valid', '%',
            'tris:ow.valid', '%', 'trisles:ow.test', '%',
            'owe-ents:ow.valid', 'owe-ents:ow.test', ]

        pd_overview = pd.DataFrame(
            data=[ds_to_row(ds) for ds in self.datasets],
            columns=cols)

        st.dataframe(pd_overview)

    # ---

    def _show_dataset_stats(self, ds: split.Dataset):
        st.write('#### Statistics')
        st.write(f'**Concept Entities:** {len(ds.concepts)}')

        st.dataframe(pd.DataFrame({
            'triples': [
                len(ds.cw_train.triples),
                len(ds.cw_valid.triples),
                len(ds.ow_valid.triples),
                len(ds.ow_test.triples),
            ], 'entities': [
                len(ds.cw_train.entities),
                len(ds.cw_valid.entities),
                len(ds.ow_valid.entities),
                len(ds.ow_test.entities),
            ], 'ow entities': [
                len(ds.cw_train.owe),
                len(ds.cw_valid.owe),
                len(ds.ow_valid.owe),
                len(ds.ow_test.owe),
            ], 'linked c': [
                len(ds.cw_train.linked_concepts),
                len(ds.cw_valid.linked_concepts),
                len(ds.ow_valid.linked_concepts),
                len(ds.ow_test.linked_concepts),
            ], 'c triples': [
                len(ds.cw_train.concept_triples),
                len(ds.cw_valid.concept_triples),
                len(ds.ow_valid.concept_triples),
                len(ds.ow_test.concept_triples),
            ]
        }, index=[
            'open world test',
            'open world validation',
            'closed world validation',
            'closed world training'], ))

        helper.legend({
            'ow entities': 'open world entities (so far unseen)',
            'linked c': 'no of concept entities present',
            'c triples': 'no of triples containing concept entities',
        })

    def _show_concept_entities(self, ds: split.Dataset):
        st.write('#### Examples')
        num_ents = st.slider(
            'Concept Entities',
            min_value=1, max_value=len(ds.concepts),
            value=10, step=1)

        ents = list(ds.concepts)[:num_ents]
        random.shuffle(ents)

        df = pd.DataFrame({
            'id': ents,
            'cw triples': [
                (len(ds.cw_train.g.find(heads={e}, tails={e})) |
                 len(ds.cw_valid.g.find(heads={e}, tails={e})))
                for e in ents
            ],
            'ow triples': [
                (len(ds.ow_valid.g.find(heads={e}, tails={e})) |
                 len(ds.ow_test.g.find(heads={e}, tails={e})))
                for e in ents
            ],
            'degree': [ds.g.nx.degree[e] for e in ents]
        }, index=(
            ds.g.source.ents[e][:25]
            for e in ents)
        )

        opts = list(df.columns)
        sorter = st.selectbox(
            label='Sort by column',
            options=range(len(opts)),
            format_func=lambda i: opts[i], )

        asc = st.radio(
            label='Ascending?',
            options=('nope', 'yep'),
            index=1,
        ) == 'yep'

        df = df.sort_values(by=opts[sorter], ascending=asc)

        st.dataframe(df)
        helper.legend({
            'cw triples': 'no of closed world triples containing the concept',
            'ow triples': 'no of open world triples containing the concept',
            'degree': 'original node degree over all triples'
        })

    def show_dataset(self):
        assert len(self.datasets) == 1
        ds = self.datasets[0]
        st.write('***')
        st.write(f'### Dataset {ds.cfg.seed}-{ds.cfg.threshold}')
        st.write(
            f'> Created: {ds.cfg.date.ctime()}\n\n'
            f'> Git Revision: {ds.cfg.git}')
        st.write('***')

        self._show_dataset_stats(ds)
        st.write('***')
        self._show_concept_entities(ds)

    def run(self):
        st.write('## HyperFB Graph Splits')

        self.load_datasets()
        # self.datasets = [self.datasets[10]]
        self.widgets = SplitWidgets.create(self.datasets)
        self.filter_datasets()

        assert len(self.datasets) > 0
        if len(self.datasets) == 1:
            self.show_dataset()
        else:
            self.show_datasets()
