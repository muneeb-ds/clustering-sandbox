import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import pandas as pd
from stqdm import stqdm
# from yellowbrick.cluster import SilhouetteVisualizer

def elbow_plot(cluster_info):
    # kmeans_info = cluster_info[cluster_info['Model']=="KMeans"]
    cluster_info = pd.DataFrame(cluster_info)
    # cluster_df = cluster_info[cluster_info['Model']=="MiniBatchKMeans"]
    # elbow_plot(cluster_df)
    figure = plt.figure(figsize=(8, 6))
    plt.plot(cluster_info['Cluster Size'], cluster_info['WCSS'], marker='o')
    plt.title('Elbow Plot for K-Means Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    st.pyplot(figure)

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
        elbow = []
        for cluster_size in stqdm(range(2, max_clusters+1)):
            kmeans_cluster_df = cluster_df.copy()
            kmeans = KMeans(n_clusters=cluster_size)
            cluster_labels = kmeans.fit_predict(kmeans_cluster_df)

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
            
            st.session_state.models['KMeans'][cluster_size] = kmeans
            elbow.append(cluster_info_dict)
            st.session_state.cluster_info.append(cluster_info_dict)
        elbow_plot(elbow)

mini_kmeans_run = st.sidebar.checkbox("MiniBatchKMeans")
if mini_kmeans_run:
    max_clusters = st.slider("Max Clusters", min_value=3, max_value=20)
    run = st.button("Run")
    if run:
        st.session_state.models['MiniBatchKMeans'] = {}
        elbow = []
        for cluster_size in stqdm(range(2, max_clusters+1)):
            kmeans_cluster_df = cluster_df.copy()
            kmeans = MiniBatchKMeans(n_clusters=cluster_size)
            cluster_labels = kmeans.fit_predict(kmeans_cluster_df)

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
            
            st.session_state.models['MiniBatchKMeans'][cluster_size] = kmeans
            elbow.append(cluster_info_dict)
            st.session_state.cluster_info.append(cluster_info_dict)
        elbow_plot(elbow)

dbscan_run = st.sidebar.checkbox("DBSCAN")
if dbscan_run:
    run = st.button("Run")
    if run:
        st.session_state.models['DBSCAN'] = {}
        dbscan = DBSCAN(n_jobs=-1)
        dbscan_labels = dbscan.fit_predict(cluster_df)
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
        
        st.session_state.models['DBSCAN'][dbscan_clusters] = dbscan
        st.session_state.cluster_info.append(cluster_info_dict)



agg_run = st.sidebar.checkbox("Agglomerative")
if agg_run:
    run = st.button("Run")
    if run:
        st.session_state.models['Agglomerative'] = {}
        agg_cluster = AgglomerativeClustering()
        agg_labels = agg_cluster.fit_predict(cluster_df)
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
        
        st.session_state.models['Agglomerative'][agg_clusters] = agg_cluster
        st.session_state.cluster_info.append(cluster_info_dict)

if st.session_state.cluster_info:
    cluster_df = pd.DataFrame(st.session_state.cluster_info)
    cluster_df = cluster_df.drop_duplicates(subset=['Model', 'Cluster Size'], keep='last')
    st.session_state.models_output = cluster_df.copy()
    st.write(st.session_state.models)
    # st.write(cluster_df)

    # elbow_plot(cluster_df)


# if col_select:
#     encoder = st.radio("Scaler", options=['MinMax', 'Standard', "Log"])

#     selected_scaled_df = scaled_df[col_select]
#     if encoder == "MinMax":
#         scaler = MinMaxScaler()
#         selected_scaled_df = scaler.fit_transform(selected_scaled_df)

#     elif encoder == "Standard":
#         scaler = StandardScaler()
#         selected_scaled_df = scaler.fit_transform(selected_scaled_df)

#     elif encoder == "Log":
#         for cols in col_select:
#             selected_scaled_df[cols] = np.log(selected_scaled_df[cols])

#     col1, col2 = st.columns([0.1,0.9])

#     with col1:
#         run_only = st.button("Run", key='scaling_run')
#     with col2:
#         run_and_persist = st.button("Run and Persist", key='scaling_persist')

#     if run_only:
#         scaled_df[col_select] = selected_scaled_df
#         # scaled_df = pd.concat([scaled_df, selected_scaled_df], axis=1)

#     if run_and_persist:
#         scaled_df[col_select] = selected_scaled_df
#         # scaled_df = pd.concat([scaled_df, selected_scaled_df], axis=1)
#         # cat_df[cat_col] = temp_df[cat_col]
#         st.session_state.updated_df = scaled_df.copy()
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write("#### Temporary DF Head")
#         st.write(scaled_df.head())

#     with col2:
#         st.write("#### Permanent DF Head")
#         st.write(st.session_state.updated_df.head())

    