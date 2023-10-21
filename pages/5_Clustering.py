import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score
import matplotlib.pyplot as plt
import pandas as pd
from stqdm import stqdm
from src.dunn import dunn_fast
from scipy.cluster.hierarchy import dendrogram
# from yellowbrick.cluster import SilhouetteVisualizer

# @st.cache_data
def cluster_fit_predict(data, model):
    cluster_labels = model.fit_predict(data)
    cluster_size = np.unique(cluster_labels).size
    st.session_state.models[model.__class__.__name__][cluster_size] = model

    return cluster_labels

# @st.cache_resource
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

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



# scaling = st.sidebar.checkbox("Feature Normalization")
# if scaling:
st.subheader("Clustering")
cluster_df = st.session_state.updated_df.copy()

# clustering_type = st.multiselect("Cluster Type", options=['KMeans', "MiniKMeans", "Agglomerative", "DBScan"])
label_options = list(cluster_df.columns) + [None]

label = st.selectbox("Label", options=label_options)

if label:
    label_series = cluster_df[label]
    cluster_df = cluster_df.drop(columns = [label])

    st.write(label_series)

if "cluster_info" not in st.session_state:
    st.session_state.cluster_info = []

if "models" not in st.session_state:
    st.session_state.models = {}

kmeans_check = st.sidebar.checkbox("KMeans")
if kmeans_check:
    max_clusters = st.slider("Max Clusters", min_value=3, max_value=20)
    run = st.button("Run")
    if run:
        if "KMeans" not in st.session_state.models:
            st.session_state.models['KMeans'] = {}
        # elbow = []
        for cluster_size in stqdm(range(2, max_clusters+1)):
            kmeans_cluster_df = cluster_df.copy()
            kmeans = KMeans(n_clusters=cluster_size)
            cluster_labels = cluster_fit_predict(kmeans_cluster_df, kmeans)

            sil_score = silhouette_score(kmeans_cluster_df, cluster_labels, sample_size=int(0.3*len(kmeans_cluster_df)))
            wcss = kmeans.inertia_
            db_index = davies_bouldin_score(kmeans_cluster_df, cluster_labels)
            ch_score = calinski_harabasz_score(kmeans_cluster_df, cluster_labels)
            # dunn_index = dunn_fast(kmeans_cluster_df, cluster_labels)
            if label:
                ari = adjusted_rand_score(label_series, cluster_labels)
                nmi = adjusted_mutual_info_score(label_series, cluster_labels)
                fmi = fowlkes_mallows_score(label_series, cluster_labels)
            else:
                ari = nmi = None
                fmi = None

            cluster_info_dict = {"Model": "KMeans", 
                                "Cluster Size": cluster_size, 
                                "WCSS": wcss,
                                "Silhouette Score": sil_score,
                                "DB Index": db_index,
                                "Calinski Harabasz Score": ch_score,
                                # "Dunn Index": dunn_index,
                                "ARI": ari,
                                "NMI": nmi,
                                "Fowlkes-mallows": fmi}
            
            # elbow.append(cluster_info_dict)
            st.session_state.cluster_info.append(cluster_info_dict)
        # st.pyplot(elbow_plot(elbow))

mini_kmeans_run = st.sidebar.checkbox("MiniBatchKMeans")
if mini_kmeans_run:
    max_clusters = st.slider("Max Clusters", min_value=3, max_value=20)
    run = st.button("Run")
    if run:
        if "MiniBatchKMeans" not in st.session_state.models:
            st.session_state.models['MiniBatchKMeans'] = {}
        # elbow = []
        for cluster_size in stqdm(range(2, max_clusters+1)):
            kmeans_cluster_df = cluster_df.copy()
            kmeans = MiniBatchKMeans(n_clusters=cluster_size)
            cluster_labels = cluster_fit_predict(kmeans_cluster_df, kmeans)

            sil_score = silhouette_score(kmeans_cluster_df, cluster_labels, sample_size=int(0.3*len(kmeans_cluster_df)))
            wcss = kmeans.inertia_
            db_index = davies_bouldin_score(kmeans_cluster_df, cluster_labels)
            ch_score = calinski_harabasz_score(kmeans_cluster_df, cluster_labels)
            if label:
                ari = adjusted_rand_score(label_series, cluster_labels)
                nmi = adjusted_mutual_info_score(label_series, cluster_labels)
                fmi = fowlkes_mallows_score(label_series, cluster_labels)
            else:
                ari = nmi = None
                fmi = None
            cluster_info_dict = {"Model": "MiniBatchKMeans", 
                                "Cluster Size": cluster_size, 
                                "WCSS": wcss,
                                "Silhouette Score": sil_score,
                                "DB Index": db_index,
                                "Calinski Harabasz Score": ch_score,
                                "ARI": ari,
                                "NMI": nmi,
                                "Fowlkes-mallows": fmi}
            
            # elbow.append(cluster_info_dict)
            st.session_state.cluster_info.append(cluster_info_dict)
        # st.pyplot(elbow_plot(elbow))

dbscan_run = st.sidebar.checkbox("DBSCAN")
if dbscan_run:
    epsilon = st.number_input("Epsilon", min_value=0.001, step=0.001, value=0.05)
    minPts = int(st.number_input("minPts", value=5))
    run = st.button("Run")
    if run:
        if "DBSCAN" not in st.session_state.models:
            st.session_state.models['DBSCAN'] = {}
        dbscan = DBSCAN(eps=epsilon, min_samples=minPts, n_jobs=-1)
        dbscan_labels = cluster_fit_predict(cluster_df, dbscan)
        dbscan_clusters = np.unique(dbscan_labels).size
        print(dbscan_clusters)
        sil_score_dbscan = silhouette_score(cluster_df, dbscan_labels, sample_size=int(0.3*len(cluster_df)))
        db_index = davies_bouldin_score(cluster_df, dbscan_labels)
        ch_score = calinski_harabasz_score(cluster_df, dbscan_labels)
        if label:
            ari = adjusted_rand_score(label_series, dbscan_labels)
            nmi = adjusted_mutual_info_score(label_series, dbscan_labels)
            fmi = fowlkes_mallows_score(label_series, dbscan_labels)
        else:
            ari = nmi = None
            fmi = None
        cluster_info_dict = {"Model": "DBSCAN", 
                            "Cluster Size": dbscan_clusters, 
                            "WCSS": None,
                            "Silhouette Score": sil_score_dbscan,
                            "DB Index": db_index,
                            "Calinski Harabasz Score": ch_score,
                            "ARI": ari,
                            "NMI": nmi,
                            "Fowlkes-mallows": fmi}
        
        st.session_state.cluster_info.append(cluster_info_dict)


agg_run = st.sidebar.checkbox("Agglomerative")
if agg_run:
    if st.checkbox("Show Dendogram"):
        _agg_cluster = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        _agg_cluster = _agg_cluster.fit(cluster_df)
        figure = plt.figure()
        plt.title("Hierarchical Clustering Dendrogram")
        plot_dendrogram(_agg_cluster, truncate_mode="level", p=5)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        st.pyplot(figure)

    n_clusters = st.slider("n_clusters", min_value=2, max_value=20)

    if st.button("Run"):
        if "AgglomerativeClustering" not in st.session_state.models:
            st.session_state.models['AgglomerativeClustering'] = {}
        agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = cluster_fit_predict(cluster_df, agg_cluster)

        agg_clusters = agg_cluster.n_clusters_

        sil_score_agg = silhouette_score(cluster_df, agg_labels, sample_size=int(0.3*len(cluster_df)))
        db_index = davies_bouldin_score(cluster_df, agg_labels)
        ch_score = calinski_harabasz_score(cluster_df, agg_labels)
        ari = nmi = None
        fmi = None
        if label:
            ari = adjusted_rand_score(label_series, agg_labels)
            nmi = adjusted_mutual_info_score(label_series, agg_labels)
            fmi = fowlkes_mallows_score(label_series, agg_labels)
        cluster_info_dict = {"Model": "AgglomerativeClustering", 
                            "Cluster Size": agg_clusters, 
                            "WCSS": None,
                            "Silhouette Score": sil_score_agg,
                            "DB Index": db_index,
                            "Calinski Harabasz Score": ch_score,
                            "ARI": ari,
                            "NMI": nmi,
                            "Fowlkes-mallows": fmi}
        
        st.session_state.cluster_info.append(cluster_info_dict)

if st.session_state.cluster_info:
    cluster_df = pd.DataFrame(st.session_state.cluster_info)
    cluster_df = cluster_df.drop_duplicates(subset=['Model', 'Cluster Size'], keep='last')
    cluster_df = cluster_df.sort_values(by = ['Model', 'Cluster Size'], ascending=True)
    st.session_state.models_output = cluster_df.copy()
    st.write(st.session_state.models)

    