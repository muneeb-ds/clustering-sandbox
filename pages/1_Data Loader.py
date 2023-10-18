import streamlit as st
import pandas as pd

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

st.subheader("Load Data")

uploaded_file = st.file_uploader("## Dataset")

if uploaded_file:
    df = read_data(uploaded_file)
    # if "updated_df" not in st.session_state:
    st.session_state.updated_df = df.copy()
    if isinstance(df, pd.DataFrame):
        st.success("Upload successful")