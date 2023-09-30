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
    # fig = plt.figure(figsize=(8, 4))
    # Set the x-axis limits
    # sns.kdeplot(data[column], shade=True)
    # plt.xlim(data[column].min(), data[column].max())
    # plt.title(f"KDE of {column}")
    st.plotly_chart(fig)


@st.cache_data
def read_data(uploaded_file):
    if "xlsx" in uploaded_file.name:
        df = pd.read_excel(uploaded_file)
    elif "csv" in uploaded_file.name:
        df = pd.read_csv(uploaded_file)
    else:
        df = None
        st.error("Please only upload CSV or XLSX files")

    return df

st.header("Clustering Sandbox")

uploaded_file = st.file_uploader("## Dataset")

if uploaded_file:
    df = read_data(uploaded_file)
    # df = pd.DataFrame({
    #   'first column': [1, 2, 3, 4],
    #   'second column': [10, 20, 30, 40]
    # })
    if not df.empty:
        st.write(df.head())
        categoricals = ['category', 'object', 'string']
        numericals = ['int']
        numerical_cols = df.select_dtypes(exclude=categoricals).columns

        x = st.selectbox("## Numerical columns ", numerical_cols)
        # y = st.selectbox("Y-axis", numerical_cols)

        # bins = st.slider(label="bin size", min_value=5, max_value=20)
        if x:
            st.write(df[x].max())
            st.write(df[x].min())
            # histogram = generate_histogram(df, x)
            # st.plotly_chart(histogram)

            generate_kde(df, x)
            # st.plotly_chart(kde)

            # st.plotly_chart(plot)