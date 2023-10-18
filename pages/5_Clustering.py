import numpy as np
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from stqdm import stqdm
# from yellowbrick.cluster import SilhouetteVisualizer

def elbow_plot(cluster_info):
    kmeans_info = cluster_info[cluster_info['Model']=="KMeans"]
    figure = plt.figure(figsize=(8, 6))
    plt.plot(kmeans_info['Cluster Size'], kmeans_info['WCSS'], marker='o')
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

cluster_info = []
# if clustering_type == "KMeans":
# number_of_clusters = st.radio("Number of Clusters", options=['Manual', 'All'])
max_clusters = st.slider("Max Clusters", min_value=3, max_value=20)
kmeans_run = st.button("KMeans")
if kmeans_run:
    for cluster_size in stqdm(range(2, max_clusters)):
        kmeans_cluster_df = cluster_df.copy()
        kmeans = KMeans(n_clusters=cluster_size)
        cluster_labels = kmeans.fit_predict(kmeans_cluster_df)

        sil_score = silhouette_score(kmeans_cluster_df, cluster_labels, sample_size=int(0.3*len(kmeans_cluster_df)))
        wcss = kmeans.inertia_

        cluster_info_dict = {"Model": "KMeans", 
                             "Cluster Size": cluster_size, 
                             "WCSS": wcss,
                             "Silhouette Score": sil_score}
        
        cluster_info.append(cluster_info_dict)

dbscan_run = st.button("DBScan")
if dbscan_run:
    dbscan = DBSCAN(n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(cluster_df)
    dbscan_clusters = np.unique(dbscan_labels).size

    sil_score_dbscan = silhouette_score(cluster_df, dbscan_labels, sample_size=int(0.3*len(cluster_df)))
    cluster_info_dict = {"Model": "DBSCAN", 
                        "Cluster Size": dbscan_clusters, 
                        "WCSS": None,
                        "Silhouette Score": sil_score_dbscan}
    cluster_info.append(cluster_info_dict)


agg_run = st.button("Agglomerative")
if agg_run:
    agg_cluster = AgglomerativeClustering().fit(cluster_df)
    agg_labels = agg_cluster.labels_
    agg_clusters = agg_cluster.n_clusters_
    sil_score_agg = silhouette_score(cluster_df, agg_labels, sample_size=int(0.3*len(cluster_df)))
    cluster_info_dict = {"Model": "Agglomerative", 
                        "Cluster Size": agg_clusters, 
                        "WCSS": None,
                        "Silhouette Score": sil_score_agg}
    cluster_info.append(cluster_info_dict)

if cluster_info:
    cluster_df = pd.DataFrame(cluster_info)
    st.write(cluster_df)
    elbow_plot(cluster_df)


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

    