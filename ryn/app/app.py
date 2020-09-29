# -*- coding: utf-8 -*-

"""

RYN STREAMLIT

to start the webserver: streamlit run ryn/app/app.py

"""
from ryn.app import kgc
from ryn.app import splits

import streamlit as st


"""

# RŶN

Open World Knowledge Graph Completion

"""


CTXS = {
    'EMBER (KGC Embeddings)': kgc.Context(),
    'SPLIT (HyperFB Datasets)': splits.Context(),
}


def run():
    key = st.sidebar.radio('Navigation', list(CTXS))
    CTXS[key].run()


if __name__ == '__main__':
    run()
