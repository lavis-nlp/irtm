# -*- coding: utf-8 -*-

import ryn
from ryn import app
from ryn.embers import keen
from ryn.common import logging

import pandas as pd
import streamlit as st

import pathlib

from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass
from collections import defaultdict

from typing import Any
from typing import Set
from typing import Dict
from typing import Tuple


log = logging.get('app.embers')


@dataclass
class EmberWidgets(app.Widgets):

    model: str
    dimensions: int
    timestamp: datetime

    @classmethod
    def create(K, models, dimensions, timestamps):

        def _reduce(
                dic: Dict[Any, Set[pathlib.Path]],
                selected_paths: Set[pathlib.Path]):

            # reduces all path sets such that only paths remain
            # that are defined in "selected_paths"
            return dict(filter(lambda t: len(t[1]), (
                (x, (paths & selected_paths) if selected_paths else paths)
                for x, paths in dic.items()
            )))

        selected_model = app.Widgets.read(
            st.sidebar.selectbox,
            'Model',
            list(models.keys()),
            str, )

        selected_paths = models.get(
            selected_model,
            set.union(*models.values()))

        dimensions = _reduce(dimensions, selected_paths)
        selected_dimensions = app.Widgets.read(
            st.sidebar.selectbox,
            'Dimensions',
            list(map(str, dimensions.keys())),
            int, )

        selected_paths &= dimensions.get(
            selected_dimensions,
            set.union(*dimensions.values()))

        timestamps = _reduce(timestamps, selected_paths)
        selected_timestamp = app.Widgets.read(
            st.sidebar.selectbox,
            'Timestamp',
            list(map(datetime.isoformat, timestamps.keys())),
            datetime.fromisoformat, )

        return EmberWidgets(
            model=selected_model,
            dimensions=selected_dimensions,
            timestamp=selected_timestamp, )


@st.cache(allow_output_mutation=True)
def load_keen(path) -> keen.Model:
    log.error('load_keen cache miss')
    return keen.Model.from_path(path)


def keen_to_row(model: keen.Model):
    kind = 'avg'

    metrics = model.results['metrics']
    hits = metrics['hits_at_k'][kind]

    return (
        model.name,
        model.dimensions,
        model.timestamp,
        metrics['mean_rank'][kind],
        metrics['mean_reciprocal_rank'][kind],
        hits['1'], hits['3'], hits['5'], hits['10'],
    )


class Context(app.Context):

    def _sort_glob(self, glob: Tuple[pathlib.Path]):
        self._data_models = defaultdict(set)
        self._data_dimensions = defaultdict(set)
        self._data_timestamps = defaultdict(set)

        for path in glob:
            model = load_keen(path)
            self._data_models[model.name].add(path)
            self._data_dimensions[model.dimensions].add(path)
            self._data_timestamps[model.timestamp].add(path)

    def _show_models(self, selection: Set[pathlib.Path]):
        cols = [
            'model', 'dims', 'created', 'MR', 'MRR',
            'hits@1', 'hits@3', 'hits@5', 'hits@10',
        ]

        df = pd.DataFrame(
            data=[keen_to_row(load_keen(p)) for p in selection],
            columns=cols)

        st.dataframe(df)

    def _show_model(self, path: pathlib.Path):
        model = load_keen(path)

        # ---

        st.write(f'## {model.name}-{model.dimensions}')
        st.write(f'Dataset: {model.metadata["dataset_name"]}')
        st.write(f'Date: {model.timestamp.strftime("%d.%m.%Y %H:%M")}')

        times = model.results['times']
        st.write(f'Training: {timedelta(seconds=times["training"])}')
        st.write(f'Evaluation: {timedelta(seconds=times["evaluation"])}')

        stopped = model.results["stopper"]["stopped"]
        st.write(f'Stopped early: {stopped}')

        # ---

        st.write('### Evaluation')
        st.write(
            'Adjusted mean rank: '
            f'{model.results["metrics"]["adjusted_mean_rank"]:3f}')
        st.dataframe(model.metrics)

        # ---

        st.write('### Training')
        st.line_chart(model.results['losses'])

        st.write('### Parameters')
        st.write(model.parameters)

    def run(self):
        st.write('## EMBERS')
        st.write('Knowledge Graph Embeddings')

        path = ryn.ENV.EMBER_DIR
        glob = tuple(path.glob('*/*'))

        st.write(f'Reading: `{path}`')

        # sets self._data*
        self._sort_glob(glob)

        # creates widgets
        self._widgets = EmberWidgets.create(
            self._data_models,
            self._data_dimensions,
            self._data_timestamps, )

        def _get(dic, key):
            return dic[key] if key else set.union(*dic.values())

        selection = (
            _get(self._data_models, self._widgets.model) &
            _get(self._data_dimensions, self._widgets.dimensions) &
            _get(self._data_timestamps, self._widgets.timestamp)
        )

        st.write('Found {n} trained{model}model(s)'.format(
            n=len(selection),
            model=f' {self._widgets.model} ' if self._widgets.model else ' '),
        )

        if len(selection) != 1:
            self._show_models(selection)
        elif len(selection) == 1:
            self._show_model(selection.pop())
        else:
            st.write('Nothing to show here...')
