import copy
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Clustering Sandbox",
    page_icon="👋",
)

st.write("# Welcome to Clustering Sandbox! 👋")

st.markdown('''
    - **Upload Your Data**: Easily upload your dataset in various formats for immediate clustering analysis.
    - **Parameter Tuning**: Fine-tune algorithm parameters to observe their impact on clustering results.
    - **Multiple Algorithms**: Run K-Means, DBScan, Fuzzy C-Means, and other clustering algorithms with ease.
    - **Dynamic Visualization**: Watch clusters form and evolve in real-time, gaining a deeper understanding of your data distribution.
         ''')
st.sidebar.success("Select a Clustering Step")