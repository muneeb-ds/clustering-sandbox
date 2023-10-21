import streamlit as st
import pandas as pd

@st.cache_data
def read_data(uploaded_file, type, delimeter=None):
    if type == "xlsx":
        df = pd.read_excel(uploaded_file)
    elif type == "csv":
        df = pd.read_csv(uploaded_file, delimiter=delimeter, engine='python')

    return df

st.subheader("Load Data")

uploaded_file = st.file_uploader("## Dataset")

if uploaded_file:
    delimeter=None
    if uploaded_file.name.endswith("csv"):
        delimeter = st.selectbox("delimeter", options=['/', ',', ':', ';', ' '])
        if st.button("Read Data"):
            df = read_data(uploaded_file, "csv", delimeter)
            if isinstance(df, pd.DataFrame):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state.clear()
                st.session_state.updated_df = df.copy()
                st.success("Upload successful")
    elif uploaded_file.name.endswith("xlsx"):
        if st.button("Read Data"):
            df = read_data(uploaded_file, "xlsx", delimeter)
            if isinstance(df, pd.DataFrame):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.session_state.clear()
                st.session_state.updated_df = df.copy()
                st.success("Upload successful")
    else:
        df = None
        st.error("Please only upload CSV or XLSX files")
    # if "updated_df" not in st.session_state: