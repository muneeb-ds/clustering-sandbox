import copy
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


@st.cache_resource
def generate_histogram(data, column):
    fig = px.histogram(data, x=column, title=f"Distribution of {column}")
    fig.update_xaxes(range=[data[column].min(), data[column].max()])
    return fig

@st.cache_resource
def generate_kde(data, column):
    hist_data = [data[column]]
    group_labels = ['distplot']
    fig = ff.create_distplot(hist_data, group_labels)
    st.plotly_chart(fig)


@st.cache_data
def read_data(uploaded_file):
    if uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = None
        st.error("Please only upload CSV or XLSX files")

    return df

def dtypes_multiselect(df: pd.DataFrame):
    dtype_cols = {}
    # with st.sidebar:
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    with col1:
        dtype_cols['int_cols'] = st.multiselect("Integer columns", options=df.columns)
    with col2:
        dtype_cols['float_cols'] = st.multiselect("Float columns", options=df.columns)
    with col3:
        dtype_cols['cat_cols'] = st.multiselect("Categorical columns", options=df.columns)
    with col4:
        dtype_cols['dt_cols'] = st.multiselect("Datetime columns", options=df.columns)

    return dtype_cols

@st.cache_data
def assign_dtypes(df:pd.DataFrame, dtype_cols:dict):
    df[dtype_cols['int_cols']] = df[dtype_cols['int_cols']].astype(int)
    df[dtype_cols['float_cols']] = df[dtype_cols['float_cols']].astype(float)
    df[dtype_cols['cat_cols']] = df[dtype_cols['cat_cols']].astype(str)
    df[dtype_cols['dt_cols']] = df[dtype_cols['dt_cols']].astype("datetime64[ns]")

    return df


st.header("Clustering Sandbox")

uploaded_file = st.file_uploader("## Dataset")

if uploaded_file:
    df = read_data(uploaded_file)

    if not df.empty:
        st.subheader("EDA")
        st.write("#### Sample Dataframe Rows:")
        st.write(df.head())
        
        if "updated_df" not in st.session_state:
            st.session_state.updated_df = df.copy()
        df = st.session_state.updated_df.copy()
        
        select_columns_checkbox = st.sidebar.checkbox("Select Useable Columns")
        if select_columns_checkbox:
            selected_columns = st.multiselect("#### Select Columns for use", options=df.columns)
            col1, col2 = st.columns([0.1,0.9])
            with col1:
                run_only = st.button("Run")
            with col2:
                run_and_persist = st.button("Run and Persist")

            if run_only:
                df = df[selected_columns].copy()
            if run_and_persist:
                df = df[selected_columns].copy()
                st.session_state.updated_df = df[selected_columns].copy()


        typecast_sidebar = st.sidebar.checkbox("Data Typecasting")
        if typecast_sidebar:
            st.write("#### Tyepcasting:")
            typecast_dict = dtypes_multiselect(df)
            st.write(df.dtypes)

            typecast_button = st.button("Assign Dtypes")
            if typecast_button:
                df = assign_dtypes(df, typecast_dict)
                st.session_state.updated_df = df.copy()
            # if "df_outliers" not in st.session_state:

        # df = st.session_state.updated_df.copy()
        st.write("#### Descriptive statistics")
        st.write(df.describe(include="all"))
        categoricals = ['category', 'object', 'string']
        numerical_cols = df.select_dtypes(include=['number']).columns
        st.session_state.numerical_cols = numerical_cols

        plot_boxplot = st.sidebar.checkbox("View BoxPlots")

        if plot_boxplot:
            df_melted = pd.melt(df[numerical_cols])
            fig = plt.figure()
            sns.boxplot(x='variable', y='value', data=df_melted)
            st.pyplot(fig)

        # if typecast_button:
            # if "df_outliers" not in st.session_state:
                # st.session_state.updated_df = st.session_state.updated_df.copy()