import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def dim_reduce(val_df, model, size, technique):
    if technique == "pca":
        tech = PCA(n_components=2)
    elif technique == "tsne":
        tech = TSNE(n_components=2)
    tech_arr = tech.fit_transform(val_df)
    tech_df = pd.DataFrame(tech_arr)
    tech_df.columns = ['Feature1', 'Feature2']

    models = st.session_state.models
    tech_df['Cluster'] = models[model][size].labels_
    return tech_df

@st.cache_resource
def plot_reduced(tech_df):
    figure = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=tech_df, x='Feature1', y='Feature2', hue='Cluster', palette='Set1')
    plt.title('Scatter Plot with Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    return figure

@st.cache_resource
def elbow_plot(cluster_info):
    # cluster_info = pd.DataFrame(cluster_info)
    figure = plt.figure(figsize=(8, 6))
    plt.plot(cluster_info['Cluster Size'], cluster_info['WCSS'], marker='o')
    plt.title('Elbow Plot for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    return figure


validation_df = st.session_state.updated_df.copy()

st.subheader("Visual Inspection")

models_out = st.session_state.models_output
st.write(models_out)

col1, col2 = st.columns(2)

with col1:
    model = st.selectbox("Models", models_out['Model'].unique())

if "kmeans" in model.lower():
    if st.checkbox("Show Elbow Plot"):
        st.pyplot(elbow_plot(models_out[models_out['Model']==model]))

# if model:
with col2:
    size = st.selectbox("Size", models_out[models_out['Model']==model]['Cluster Size'].unique())


dim_red = st.radio("Dimensionality Reduction", options=['PCA', 't-SNE'])

if dim_red == "PCA":
    reduced_df = dim_reduce(validation_df, model, size, "pca")

if dim_red == "t-SNE":
    reduced_df = dim_reduce(validation_df, model, size, "tsne")

if st.button("Show Plot"):
    st.pyplot(plot_reduced(reduced_df))