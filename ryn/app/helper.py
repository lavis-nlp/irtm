# -*- coding: utf-8 -*-

import streamlit as st


def legend(dic):
    st.write('_Legend:_')
    st.write('\n\n'.join(f'> **{k}:** {v}' for k, v in dic.items()))
