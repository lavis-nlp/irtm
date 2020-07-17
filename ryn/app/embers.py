# -*- coding: utf-8 -*-

from ryn import app

import streamlit as st


class Context(app.Context):

    def run(self):
        st.write('## EMBERS')
        st.write('Knowledge Graph Embeddings')
        st.write("coming soon...")
