import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def plot_reduced(val_df, technique):
    if technique == "pca":
        tech = PCA(n_components=2)
    elif technique == "tsne":
        tech = TSNE(n_components=2)
    if st.button("Show Plot"):
        tech_arr = tech.fit_transform(val_df)
        tech_df = pd.DataFrame(tech_arr)
        tech_df.columns = ['Feature1', 'Feature2']

        models = st.session_state.models
        tech_df['Cluster'] = models[model][size].labels_

        figure = plt.figure(figsize=(8, 6))
        sns.scatterplot(data=tech_df, x='Feature1', y='Feature2', hue='Cluster', palette='Set1')
        plt.title('Scatter Plot with Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        st.pyplot(figure)


validation_df = st.session_state.updated_df.copy()

st.subheader("Visual Inspection")

models_out = st.session_state.models_output
st.write(models_out)

col1, col2 = st.columns(2)

with col1:
    model = st.selectbox("Models", models_out['Model'].unique())

# if model:
with col2:
    size = st.selectbox("Size", models_out[models_out['Model']==model]['Cluster Size'].unique())


dim_red = st.radio("Dimensionality Reduction", options=['PCA', 't-SNE'])

if dim_red == "PCA":
    plot_reduced(validation_df, "pca")

if dim_red == "t-SNE":
    plot_reduced(validation_df, "tsne")