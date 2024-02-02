import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
from matplotlib.patches import Ellipse
from scipy import linalg

label = st.session_state.label

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
    if model == "GaussianMixture" or model == "FCM":
        feature_set = st.session_state.updated_df.loc[:, st.session_state.updated_df.columns.difference([label])]
        tech_df['Cluster'] = models[model][size].predict(feature_set.to_numpy())
    else:
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

@st.cache_resource
def aic_bic_plot(X):
    gmm_models = st.session_state.models["GaussianMixture"]
    max_component = max(gmm_models.keys())
    n_components = np.arange(2, max_component+1)
    fig = plt.figure()
    plt.plot(n_components, [m.bic(X) for m in gmm_models.values()], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in gmm_models.values()], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    return fig

@st.cache_resource
def pc_vs_pec_plot():
    fcm_models = st.session_state.models['FCM']
    max_component = max(fcm_models.keys())
    n_components = np.arange(2, max_component+1)
    fig = plt.figure()
    plt.plot(n_components, [m.partition_coefficient for m in fcm_models.values()], label="Partition Coefficient")
    plt.plot(n_components, [m.partition_entropy_coefficient for m in fcm_models.values()], label="Partition Entropy Coefficient")
    plt.legend(loc="best")
    plt.xlabel('n_clusters')
    return fig

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

@st.cache_resource
def plot_gmm(df, cluster_size, title='GMM'):
    gmm = st.session_state.models['GaussianMixture'][cluster_size]
    X = df[['Feature1', 'Feature2']].to_numpy()
    Y = df['Cluster'].to_numpy()
    means = gmm.means_
    covariances = gmm.covariances_
    figure = plt.figure()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_alpha(0.5)
        plt.gca().add_artist(ell)

    # plt.xlim(-9.0, 5.0)
    # plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

    return figure

@st.cache_resource
def plot_fcm(df, cluster_size):
    model = st.session_state.models['FCM'][cluster_size]
    X = df[['Feature1', 'Feature2']].to_numpy()
    Y = df['Cluster'].to_numpy()
    # num_clusters = len(n_clusters_list)
    # rows = int(np.ceil(np.sqrt(num_clusters)))
    # cols = int(np.ceil(num_clusters / rows))
    fig = plt.figure()
    # for n_clusters, model, axe in zip(n_clusters_list, models, axes.ravel()):
        # get validation metrics
    pc = model.partition_coefficient
    pec = model.partition_entropy_coefficient
    
    fcm_centers = model.centers
    # Y = model.predict(X)
    # plot result
    plt.scatter(X[:,0], X[:,1], c=Y, alpha=.1)
    plt.scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='black')
    plt.title(f'PC = {pc:.3f}, PEC = {pec:.3f}')
    return fig

validation_df = st.session_state.updated_df.copy()

st.subheader("Visual Inspection")

models_out = st.session_state.models_output
st.write(models_out)

# col1, col2 = st.columns(2)

# st.write(max(st.session_state.models["GaussianMixture"].keys()))

# with col1:
model = st.selectbox("Models", models_out['Model'].unique())

if "kmeans" in model.lower():
    if st.checkbox("Show Elbow Plot"):
        st.pyplot(elbow_plot(models_out[models_out['Model']==model]))

if "gaussianmixture" in model.lower():
    if st.checkbox("Show AIC/BIC Plot"):
        st.pyplot(aic_bic_plot(validation_df.loc[:, validation_df.columns.difference([label])].to_numpy()))

if "fcm" in model.lower():
    if st.checkbox("Show PC vs PEC Plot"):
        st.pyplot(pc_vs_pec_plot())
# if model:
# with col2:
size = st.selectbox("Size", models_out[models_out['Model']==model]['Cluster Size'].unique())


dim_red = st.radio("Dimensionality Reduction", options=['PCA', 't-SNE'])

if dim_red == "PCA":
    reduced_df = dim_reduce(validation_df, model, size, "pca")

if dim_red == "t-SNE":
    reduced_df = dim_reduce(validation_df, model, size, "tsne")

if st.button("Show Plot"):
    if "gaussianmixture" in model.lower():
        st.pyplot(plot_gmm(reduced_df, size))
    elif "fcm" in model.lower():
        st.pyplot(plot_fcm(reduced_df, size))
    else:
        st.pyplot(plot_reduced(reduced_df))