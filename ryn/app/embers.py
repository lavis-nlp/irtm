# -*- coding: utf-8 -*-

from ryn import app

from dataclasses import dataclass

import streamlit as st


HEADER = """

## EMBERS

Knowledge Graph Embeddings

"""


@dataclass
class Context(app.Context):

    def run(self):
        st.write("coming soon...")
