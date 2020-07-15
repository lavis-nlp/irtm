# -*- coding: utf-8 -*-

"""

RYN STREAMLIT

to start the webserver: streamlit run ryn_app.py

"""
from ryn.app import splits
from ryn.app import embers

import streamlit as st


"""

# RŶN

Open World Knowledge Graph Completion

"""


CTXS = {
    'KGC Embeddings': embers.Context(),
    'HyperFB Datasets': splits.Context(),
}


def run():
    key = st.sidebar.radio('Navigation', list(CTXS))
    CTXS[key].run()


run()
