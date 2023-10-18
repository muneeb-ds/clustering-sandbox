import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import pandas as pd
from stqdm import stqdm
# from yellowbrick.cluster import SilhouetteVisualizer

@st.cache_data
def cluster_fit_predict(data, model, cluster_size=None):
    match model:
        case "KMeans":
            method = KMeans(n_clusters=cluster_size)
            cluster_labels = method.fit_predict(data)
        case "MiniBatchKMeans":
            method = MiniBatchKMeans(n_clusters=cluster_size)
            cluster_labels = method.fit_predict(data)
        case "DBSCAN":
            method = DBSCAN(n_jobs=-1)
            cluster_labels = method.fit_predict(data)
            cluster_size = np.unique(cluster_labels).size
        case "Agglomerative":
            method = AgglomerativeClustering()
            cluster_labels = method.fit_predict(data)
            cluster_size = method.n_clusters_

    st.session_state.models[model][cluster_size] = method

    return method, cluster_labels

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
        st.session_state.models['KMeans'] = {}
        # elbow = []
        for cluster_size in stqdm(range(2, max_clusters+1)):
            kmeans_cluster_df = cluster_df.copy()
            kmeans, cluster_labels = cluster_fit_predict(kmeans_cluster_df, "KMeans", cluster_size)

            sil_score = silhouette_score(kmeans_cluster_df, cluster_labels, sample_size=int(0.3*len(kmeans_cluster_df)))
            wcss = kmeans.inertia_
            db_index = davies_bouldin_score(kmeans_cluster_df, cluster_labels)
            ch_score = calinski_harabasz_score(kmeans_cluster_df, cluster_labels)
            if label:
                ari = adjusted_rand_score(label_series, cluster_labels)
                nmi = adjusted_mutual_info_score(label_series, cluster_labels)
            else:
                ari = nmi = None

            cluster_info_dict = {"Model": "KMeans", 
                                "Cluster Size": cluster_size, 
                                "WCSS": wcss,
                                "Silhouette Score": sil_score,
                                "DB Index": db_index,
                                "Calinski Harabasz Score": ch_score,
                                "ARI": ari,
                                "NMI": nmi}
            
            # elbow.append(cluster_info_dict)
            st.session_state.cluster_info.append(cluster_info_dict)
        # st.pyplot(elbow_plot(elbow))

mini_kmeans_run = st.sidebar.checkbox("MiniBatchKMeans")
if mini_kmeans_run:
    max_clusters = st.slider("Max Clusters", min_value=3, max_value=20)
    run = st.button("Run")
    if run:
        st.session_state.models['MiniBatchKMeans'] = {}
        # elbow = []
        for cluster_size in stqdm(range(2, max_clusters+1)):
            kmeans_cluster_df = cluster_df.copy()
            kmeans, cluster_labels = cluster_fit_predict(kmeans_cluster_df, "MiniBatchKMeans", cluster_size)

            sil_score = silhouette_score(kmeans_cluster_df, cluster_labels, sample_size=int(0.3*len(kmeans_cluster_df)))
            wcss = kmeans.inertia_
            db_index = davies_bouldin_score(kmeans_cluster_df, cluster_labels)
            ch_score = calinski_harabasz_score(kmeans_cluster_df, cluster_labels)
            if label:
                ari = adjusted_rand_score(label_series, cluster_labels)
                nmi = adjusted_mutual_info_score(label_series, cluster_labels)
            else:
                ari = nmi = None
            cluster_info_dict = {"Model": "MiniBatchKMeans", 
                                "Cluster Size": cluster_size, 
                                "WCSS": wcss,
                                "Silhouette Score": sil_score,
                                "DB Index": db_index,
                                "Calinski Harabasz Score": ch_score,
                                "ARI": ari,
                                "NMI": nmi}
            
            # elbow.append(cluster_info_dict)
            st.session_state.cluster_info.append(cluster_info_dict)
        # st.pyplot(elbow_plot(elbow))

dbscan_run = st.sidebar.checkbox("DBSCAN")
if dbscan_run:
    run = st.button("Run")
    if run:
        st.session_state.models['DBSCAN'] = {}
        dbscan, dbscan_labels = cluster_fit_predict(kmeans_cluster_df, "DBSCAN", cluster_size)
        dbscan_clusters = np.unique(dbscan_labels).size

        sil_score_dbscan = silhouette_score(cluster_df, dbscan_labels, sample_size=int(0.3*len(cluster_df)))
        db_index = davies_bouldin_score(cluster_df, dbscan_labels)
        ch_score = calinski_harabasz_score(cluster_df, dbscan_labels)
        if label:
            ari = adjusted_rand_score(label_series, dbscan_labels)
            nmi = adjusted_mutual_info_score(label_series, dbscan_labels)
        else:
            ari = nmi = None
        cluster_info_dict = {"Model": "DBSCAN", 
                            "Cluster Size": dbscan_clusters, 
                            "WCSS": None,
                            "Silhouette Score": sil_score_dbscan,
                            "DB Index": db_index,
                            "Calinski Harabasz Score": ch_score,
                            "ARI": ari,
                            "NMI": nmi}
        
        st.session_state.cluster_info.append(cluster_info_dict)


agg_run = st.sidebar.checkbox("Agglomerative")
if agg_run:
    run = st.button("Run")
    if run:
        st.session_state.models['Agglomerative'] = {}
        agg_cluster, agg_labels = cluster_fit_predict(kmeans_cluster_df, "Agglomerative", cluster_size)
        agg_clusters = agg_cluster.n_clusters_

        sil_score_agg = silhouette_score(cluster_df, agg_labels, sample_size=int(0.3*len(cluster_df)))
        db_index = davies_bouldin_score(cluster_df, agg_labels)
        ch_score = calinski_harabasz_score(cluster_df, agg_labels)
        if label:
            ari = adjusted_rand_score(label_series, agg_labels)
            nmi = adjusted_mutual_info_score(label_series, agg_labels)
        else:
            ari = nmi = None
        cluster_info_dict = {"Model": "Agglomerative", 
                            "Cluster Size": agg_clusters, 
                            "WCSS": None,
                            "Silhouette Score": sil_score_agg,
                            "DB Index": db_index,
                            "Calinski Harabasz Score": ch_score,
                            "ARI": ari,
                            "NMI": nmi}
        
        st.session_state.cluster_info.append(cluster_info_dict)

if st.session_state.cluster_info:
    cluster_df = pd.DataFrame(st.session_state.cluster_info)
    cluster_df = cluster_df.drop_duplicates(subset=['Model', 'Cluster Size'], keep='last')
    st.session_state.models_output = cluster_df.copy()
    st.write(st.session_state.models)

    