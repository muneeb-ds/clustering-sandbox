import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from collections import defaultdict

st.subheader("Feature Selection")

fe_df = st.session_state.updated_df.copy()

@st.cache_resource
def plot_corr_map(df):
    fig = plt.figure(figsize=(30,20))
    sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap = 'YlGnBu')
    return fig

@st.cache_resource
def create_pairwise_plots(dataframe, hue=None):
    # fig = plt.figure(figsize=(30,20))
    # sns.set(style="ticks")
    fig = sns.pairplot(dataframe, hue=hue, palette='husl', markers='o')
    return fig

@st.cache_data
def threshold_categorical_values(df,column, type='percentile', threshold=100):
    # for column in df.select_dtypes(include=['category', 'object', 'string']).columns:
    value_counts = df[column].value_counts()
    total_values = 1
    if type == 'percentile':
        total_values = len(df[column])

    values_to_replace = value_counts[(value_counts/total_values) < threshold].index
    df[column] = df[column].replace(values_to_replace, 'others')
    return df

@st.cache_resource
def plot_value_counts(df, cat_col):
    value_counts = df[cat_col].value_counts()
    fig = plt.figure()
    plt.bar(value_counts.index, value_counts)
    plt.xlabel(cat_col)
    plt.ylabel('Count')
    plt.title(f'Original Value Counts - {cat_col}')
    plt.xticks(rotation=45, ha='right')
    return fig


if st.sidebar.checkbox("Categorical Thresholding"):
    st.subheader("Categorical Thresholding")
    cat_df = st.session_state.updated_df.copy()
    categorical_cols = cat_df.select_dtypes(include=['category', 'object', 'string']).columns
    cat_col = st.selectbox("Select Categorical Column for Value Counts Graph", categorical_cols)
    if cat_col:
        if st.checkbox("Show Bar Plot"):
            st.pyplot(plot_value_counts(cat_df, cat_col))
        else:
            st.write(cat_df[cat_col].value_counts())

        type = st.selectbox("Threshold Type", options=['Percentage', 'Count'])
        threshold = st.number_input("Threshold Value", min_value=1, max_value=100)



        col1, col2 = st.columns([0.1,0.9])
        with col1:
            run_only = st.button("Run", key='thresholding_run')
        with col2:
            run_and_persist = st.button("Run and Persist", key='thresholding_persist')

        temp_df = threshold_categorical_values(cat_df, cat_col, type = type, threshold=threshold)


        if run_only:
            cat_df = temp_df.copy()

        if run_and_persist:
            st.session_state.updated_df = temp_df.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Temporary DF Head")
            st.write(cat_df.head())

        with col2:
            st.write("#### Permanent DF Head")
            st.write(st.session_state.updated_df.head())

if st.sidebar.checkbox("Categorical Encoding"):
    st.subheader("Categorical Encoding")
    cat_df = st.session_state.updated_df.copy()
    categorical_cols = cat_df.select_dtypes(include=['category', 'object', 'string']).columns
    # cat_df = cat_df[categorical_cols]

    cat_col = st.multiselect("Categorical Columns", options=categorical_cols)
    all_options = st.checkbox("Select All Columns")
    if all_options:
        cat_col = list(categorical_cols)

    if cat_col:

        encoder = st.radio("Encoder", options=['Label', 'OneHot'])

        if encoder == "Label":
            label_dict = defaultdict(LabelEncoder)
            temp_df = cat_df[cat_col].apply(lambda x: label_dict[x.name].fit_transform(x))
        
        elif encoder == "OneHot":
            ohe = OneHotEncoder()
            array_hot_encoded = ohe.fit_transform(cat_df[cat_col])
            temp_df = pd.DataFrame(array_hot_encoded.toarray())
            temp_df.columns = ohe.get_feature_names_out()

        col1, col2 = st.columns([0.1,0.9])
        with col1:
            run_only = st.button("Run", key='encoding_run')
        with col2:
            run_and_persist = st.button("Run and Persist", key='encoding_persist')

        if run_only:
            cat_df.drop(columns = cat_col, inplace=True)
            cat_df = pd.concat([cat_df, temp_df], axis=1)

        if run_and_persist:
            cat_df.drop(columns = cat_col, inplace=True)
            cat_df = pd.concat([cat_df, temp_df], axis=1)
            # cat_df[cat_col] = temp_df[cat_col]
            st.session_state.updated_df = cat_df.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Temporary DF Head")
            st.write(cat_df.head())

        with col2:
            st.write("#### Permanent DF Head")
            st.write(st.session_state.updated_df.head())


scaling = st.sidebar.checkbox("Feature Normalization")
if scaling:
    st.subheader("Feature Scaling")
    scaled_df = st.session_state.updated_df.copy()
    numerical_cols = scaled_df.columns
    col_select = st.multiselect("Numerical Columns", options=numerical_cols)
    all_options = st.checkbox("Select All Columns")
    if all_options:
        col_select = list(numerical_cols)

    if col_select:
        encoder = st.radio("Scaler", options=['MinMax', 'Standard', "Log"])

        selected_scaled_df = scaled_df[col_select]
        if encoder == "MinMax":
            scaler = MinMaxScaler()
            selected_scaled_df = scaler.fit_transform(selected_scaled_df)

        elif encoder == "Standard":
            scaler = StandardScaler()
            selected_scaled_df = scaler.fit_transform(selected_scaled_df)

        elif encoder == "Log":
            for cols in col_select:
                selected_scaled_df[cols] = np.log(selected_scaled_df[cols])

        col1, col2 = st.columns([0.1,0.9])

        with col1:
            run_only = st.button("Run", key='scaling_run')
        with col2:
            run_and_persist = st.button("Run and Persist", key='scaling_persist')

        if run_only:
            scaled_df[col_select] = selected_scaled_df
            # scaled_df = pd.concat([scaled_df, selected_scaled_df], axis=1)

        if run_and_persist:
            scaled_df[col_select] = selected_scaled_df
            # scaled_df = pd.concat([scaled_df, selected_scaled_df], axis=1)
            # cat_df[cat_col] = temp_df[cat_col]
            st.session_state.updated_df = scaled_df.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Temporary DF Head")
            st.write(scaled_df.head())

        with col2:
            st.write("#### Permanent DF Head")
            st.write(st.session_state.updated_df.head())

if st.sidebar.checkbox("Pairwise Plots"):
    st.write("#### Pairwise Plots")
    label = st.session_state.label
    st.pyplot(create_pairwise_plots(fe_df, label))

if st.sidebar.checkbox("Correlation Map"):
    st.write("#### Correlation Map")
    st.write(fe_df.corr())

    st.pyplot(plot_corr_map(fe_df))

# if st.sidebar.checkbox("Select K Best"):
#     st.subheader("Select K Best Features")
#     kbest_df = st.session_state.updated_df.copy()
#     kbest = SelectKBest(mutual_info_classif, k='all')
#     X = kbest.fit_transform(kbest_df)
#     feature_names = kbest_df.columns[X.get_support()]
#     feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': kbest.scores_})
#     st.write(feature_scores.sort_values("Score", ascending=False))

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