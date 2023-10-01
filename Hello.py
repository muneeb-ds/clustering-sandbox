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
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Clustering Sandbox! 👋")

st.sidebar.success("Select a Clustering Step")