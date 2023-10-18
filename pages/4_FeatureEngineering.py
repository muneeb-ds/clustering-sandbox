import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("Feature Selection")

fe_df = st.session_state.updated_df.copy()

st.write("#### Correlation Map")
st.write(fe_df.corr())
fig = plt.figure()
sns.heatmap(fe_df.corr(), annot=True, vmin=-1, vmax=1, cmap = 'YlGnBu')
st.pyplot(fig)


st.subheader("Clustering Features to Use")

# cols = st.multiselect("Features", options= fe_df.columns)
selected_columns = st.multiselect("#### Features", options=fe_df.columns)
if selected_columns:

    col1, col2 = st.columns([0.1,0.9])
    with col1:
        run_only = st.button("Run", key="feature_select_run")
    with col2:
        run_and_persist = st.button("Run and Persist", key="feature_select_persist")

    if run_only:
        fe_df = fe_df[selected_columns].copy()
    if run_and_persist:
        fe_df = fe_df[selected_columns].copy()
        st.session_state.updated_df = fe_df[selected_columns].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Temporary DF Columns")
        st.write(fe_df.columns)

    with col2:
        st.write("#### Permanent DF Columns")
        st.write(st.session_state.updated_df.columns)