import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from collections import defaultdict
from feature_engine.datetime import DatetimeFeatures


@st.cache_data
def cap_outliers(data, col, min_cap, max_cap):
    data[col] = np.where(data[col]>max_cap, max_cap, data[col])
    data[col] = np.where(data[col]<min_cap, min_cap, data[col])

    return data

@st.cache_data
def remove_percentile_outliers(data, col, lower_quantile, upper_quantile):
    return data[(data[col] > lower_quantile) & 
                        (data[col] < upper_quantile)]


@st.cache_data
def copy_updated_df():
    st.session_state.updated_df = st.session_state.temp_df.copy()


def capping_trigger(df_outliers, col, min_cap, max_cap):
    df_outliers = cap_outliers(df_outliers, col, min_cap, max_cap)
    st.write(df_outliers.head())
    st.write(f"#### {col} Boxplot after outlier removal")
    fig = plt.figure()
    sns.boxplot(data=df_outliers[col], orient="horizontal")
    st.pyplot(fig)

    return df_outliers


def percentile_trigger(df_outliers, col, lower_quantile, upper_quantile):
    df_outliers = remove_percentile_outliers(df_outliers, col, lower_quantile, upper_quantile)
    st.write(df_outliers.head())
    st.write(f"#### {col} Boxplot after outlier removal")
    fig = plt.figure()
    sns.boxplot(data=df_outliers[col], orient="horizontal")
    st.pyplot(fig)

    return df_outliers

def fix_missing_vals(missing_df, method, custom = None):
    match method:
        case 'drop':
            missing_df = missing_df.dropna(axis=0, how = "any", subset = selected_cols)

        case 'custom text':
            missing_df[selected_cols] = missing_df[selected_cols].fillna(custom)

        case 'custom value':
            missing_df[selected_cols] = missing_df[selected_cols].fillna(value)

        case 'ffill' | 'bfill':
            missing_df[selected_cols] = missing_df[selected_cols].fillna(method=method)

        case 'mean' | "median" | "most_frequent":
            imputer = SimpleImputer(missing_values=np.nan, strategy=method)
            imputer = imputer.fit(missing_df[selected_cols])

            missing_df[selected_cols] = imputer.transform(missing_df[selected_cols]) 
        
        case 'linear interpolation':
            missing_df[selected_cols] = missing_df[selected_cols].interpolate(method ='linear')

        case 'quadratic interpolation':
            missing_df[selected_cols] = missing_df[selected_cols].interpolate(method ='quadratic')

    return missing_df




st.subheader("Pre-Processing")

remove_outliers = st.sidebar.checkbox(label="Remove Outliers")
if remove_outliers:
    st.subheader(" Remove Outliers:")
    numerical_cols = st.session_state.numerical_cols
    col = st.selectbox(label="Remove outlier for", options=numerical_cols, key='outlier_cols')
    if col:
        df_outliers = st.session_state.updated_df.copy()
        method = st.radio("Outlier removal method", options=['capping', 'percentile'])

        if method == "capping":

            st.write(f"Capping for {col}")
            min_cap = st.number_input(label="Min Cap")
            max_cap = st.number_input(label="Max Cap")

            col1, col2 = st.columns([0.1,0.9])
            with col1:
                run_only = st.button("Run")
            with col2:
                run_and_persist = st.button("Run and Persist")

            if run_only:
                df_outliers = capping_trigger(df_outliers, col, min_cap, max_cap)
                # persist = st.button("Persist change", key="persist", on_click=copy_updated_df)
            
            if run_and_persist:
                df_outliers = capping_trigger(df_outliers, col, min_cap, max_cap)
                st.session_state.updated_df = df_outliers.copy()

        elif method == "percentile":

            min_pct = st.number_input(label="Min Percentile", min_value=0.00, max_value=1.00, step=0.01)
            max_pct = st.number_input(label="Max Percentile", min_value=0.00, max_value=1.00, step=0.01)

            lower_quantile = df_outliers[col].quantile(min_pct)
            upper_quantile = df_outliers[col].quantile(max_pct)

            col1, col2 = st.columns([0.1,0.9])
            with col1:
                run_only = st.button("Run")
            with col2:
                run_and_persist = st.button("Run and Persist")

            if run_only:
                df_outliers = percentile_trigger(df_outliers, col, lower_quantile, upper_quantile)
                # persist = st.button("Persist change", on_click=copy_updated_df)

            if run_and_persist:
                df_outliers = percentile_trigger(df_outliers, col, lower_quantile, upper_quantile)
                st.session_state.updated_df = df_outliers.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Temporary DF Stats")
            st.write(df_outliers.describe(include="number"))

        with col2:
            st.write("#### Permanent DF Stats")
            st.write(st.session_state.updated_df.describe(include="number"))


missing_vals = st.sidebar.checkbox("Fix Missing Values")
if missing_vals:
    missing_df = st.session_state.updated_df.copy()
    st.subheader("Impute Missing Values")

    total_missing = missing_df.isnull().sum().sum()
    st.write(f"Total Missing values: {total_missing}")
    st.write("Missing values for each column:")
    st.write(missing_df.isnull().sum())

    selected_cols = st.multiselect("Columns w/ Missing Vals", options=missing_df.columns)
    all_options = st.checkbox("Select All Missing Columns")
    if all_options:
        selected_cols = list(missing_df.columns)
    
    method = st.radio("Method", options=['drop','custom text','custom value', 'ffill', 'bfill','mean', 'median','most_frequent', 'linear interpolation', 'quadratic interpolation'])
    custom = None
    if method == "custom text":
        custom = st.text_input("Custom Text")
    if method == "custom value":
        custom = st.number_input("Custom Number")

    col1, col2 = st.columns([0.1,0.9])
    with col1:
        run_only = st.button("Run", key='missing_val_run')
    with col2:
        run_and_persist = st.button("Run and Persist", key='missing_val_persist')

    if selected_cols:
        if run_only:
            missing_df = fix_missing_vals(missing_df, method, custom)
        if run_and_persist:
            missing_df = fix_missing_vals(missing_df, method, custom)
            st.session_state.updated_df = missing_df.copy()

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Temporary DF Stats")
            st.write(missing_df.describe(include="all"))

        with col2:
            st.write("#### Permanent DF Stats")
            st.write(st.session_state.updated_df.describe(include="all"))


dt_prep = st.sidebar.checkbox("Datetime Prep")
if dt_prep:
    st.subheader("Datetime Feature Engineering")
    dt_df = st.session_state.updated_df.copy()
    dt_columns = dt_df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
    col_select = st.multiselect("Datetime Columns", options=dt_columns)
    if col_select:
        original_cols = dt_df.columns
        dt_feature_eng = DatetimeFeatures(variables=col_select, features_to_extract='all')
        dt_df_transf = dt_feature_eng.fit_transform(dt_df)
        feature_names = dt_df_transf[dt_df_transf.columns.difference(original_cols)].columns
        # feature_names = dt_feature_eng.get_feature_names_out()
        dt_variables = st.multiselect("Datetime Features to keep", options=feature_names)


        col1, col2 = st.columns([0.1,0.9])

        with col1:
            run_only = st.button("Run", key='scaling_run')
        with col2:
            run_and_persist = st.button("Run and Persist", key='scaling_persist')

        if run_only:
            dt_df_transf = dt_df_transf[dt_variables].astype(int)
            dt_df[dt_variables] = dt_df_transf[dt_variables]
            dt_df = dt_df.drop(columns=col_select)

        if run_and_persist:
            dt_df_transf = dt_df_transf[dt_variables]
            dt_df[dt_variables] = dt_df_transf[dt_variables]
            dt_df = dt_df.drop(columns=col_select)
            st.session_state.updated_df = dt_df.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Temporary DF Head")
            st.write(dt_df.head())

        with col2:
            st.write("#### Permanent DF Head")
            st.write(st.session_state.updated_df.head())

duplicating = st.sidebar.checkbox("Drop Duplicates")
if duplicating:
    st.subheader("Drop Duplicates")
    duplicated_df = st.session_state.updated_df.copy()
    # numerical_cols = duplicated_df.columns

    col1, col2 = st.columns([0.1,0.9])

    with col1:
        run_only = st.button("Run", key='duplicated_run')
    with col2:
        run_and_persist = st.button("Run and Persist", key='duplicated_persist')

    if run_only:
        duplicated_df = duplicated_df.drop_duplicates()
        # scaled_df = pd.concat([scaled_df, selected_scaled_df], axis=1)

    if run_and_persist:
        duplicated_df = duplicated_df.drop_duplicates()
        # scaled_df = pd.concat([scaled_df, selected_scaled_df], axis=1)
        # cat_df[cat_col] = temp_df[cat_col]
        st.session_state.updated_df = duplicated_df
        
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Temporary DF Size")
        st.write(duplicated_df.size)

    with col2:
        st.write("#### Permanent DF Size")
        st.write(st.session_state.updated_df.size)
