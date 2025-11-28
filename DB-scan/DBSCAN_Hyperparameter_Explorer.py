import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="DBSCAN Explorer", layout="wide")

# Title
st.title("🔍 DBSCAN Hyperparameter Explorer")
st.markdown("Explore how different hyperparameters affect DBSCAN clustering on the Wholesale Customers dataset")

# Load and prepare data
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
    df_full = pd.read_csv(url)
    df = df_full[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].copy()
    
    # Normalize
    X = StandardScaler().fit_transform(df)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    return df_full, df, X, X_2d, pca

df_full, df, X, X_2d, pca = load_data()

# Sidebar for hyperparameters
st.sidebar.header("Hyperparameters")

# Epsilon (eps)
eps = st.sidebar.slider(
    "Epsilon (eps)",
    min_value=0.1,
    max_value=3.0,
    value=0.6,
    step=0.1,
    help="Maximum distance between two samples to be considered neighbors"
)

# Minimum samples
min_samples = st.sidebar.slider(
    "Minimum Samples",
    min_value=2,
    max_value=20,
    value=5,
    help="Number of samples in a neighborhood for a point to be considered a core point"
)

# Metric
metric = st.sidebar.selectbox(
    "Distance Metric",
    options=["euclidean", "manhattan", "cosine"],
    help="Distance metric used to calculate neighborhood"
)

# PCA options
show_pca_variance = st.sidebar.checkbox("Show PCA Explained Variance", value=False)
show_metrics = st.sidebar.checkbox("Show Clustering Metrics", value=True)

# Main content area
col1, col2 = st.columns(2)

# Perform clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
labels = dbscan.fit_predict(X)

# Calculate metrics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
n_clustered = len(labels) - n_noise

# Column 1: Clustering visualization
with col1:
    st.subheader("Clustering Results")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each cluster separately
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in gray
            mask = labels == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                      c='gray', marker='x', s=50, alpha=0.5, label='Noise')
        else:
            mask = labels == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                      c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
    
    ax.set_title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    st.pyplot(fig)
    plt.close()
    
    # Cluster statistics
    st.subheader("Cluster Statistics")
    st.metric("Number of Clusters", n_clusters)
    st.metric("Noise Points", f"{n_noise} ({n_noise/len(labels)*100:.1f}%)")
    st.metric("Clustered Points", f"{n_clustered} ({n_clustered/len(labels)*100:.1f}%)")
    
    unique_labels_sorted = sorted([l for l in set(labels) if l != -1])
    if unique_labels_sorted:
        counts = [list(labels).count(l) for l in unique_labels_sorted]
        cluster_df = pd.DataFrame({
            'Cluster': unique_labels_sorted,
            'Count': counts,
            'Percentage': [(c / len(labels) * 100) for c in counts]
        })
        cluster_df['Percentage'] = cluster_df['Percentage'].round(2)
        st.dataframe(cluster_df, use_container_width=True)

# Column 2: Additional visualizations and metrics
with col2:
    # Clustering metrics
    if show_metrics and n_clusters > 1 and n_clustered > 0:
        st.subheader("Clustering Quality Metrics")
        
        # Filter out noise points for metrics
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            X_non_noise = X[non_noise_mask]
            labels_non_noise = labels[non_noise_mask]
            
            try:
                sil = silhouette_score(X_non_noise, labels_non_noise)
                dbi = davies_bouldin_score(X_non_noise, labels_non_noise)
                ch = calinski_harabasz_score(X_non_noise, labels_non_noise)
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz'],
                    'Value': [f"{sil:.4f}", f"{dbi:.4f}", f"{ch:.2f}"],
                    'Note': ['Higher is better', 'Lower is better', 'Higher is better']
                })
                st.dataframe(metrics_df, use_container_width=True)
                
                st.caption(f"*Metrics calculated on {n_clustered} non-noise points*")
            except:
                st.warning("Unable to calculate metrics with current parameters")
    elif n_clusters <= 1:
        st.subheader("Clustering Quality Metrics")
        st.warning(f"Only {n_clusters} cluster(s) found. Adjust parameters to find more clusters.")

    # PCA variance
    if show_pca_variance:
        st.subheader("PCA Explained Variance")
        
        pca_full = PCA()
        pca_full.fit(X)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
               pca_full.explained_variance_ratio_, alpha=0.7)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained")
        ax.set_title("PCA Explained Variance by Component")
        st.pyplot(fig)
        plt.close()
        
        st.write(f"First 2 components explain: {sum(pca_full.explained_variance_ratio_[:2]):.2%} of variance")

# Full width section: Comparison by original features
st.markdown("---")
st.subheader("Cluster Analysis by Original Features")

# Add cluster labels to dataframe
df_with_labels = df.copy()
df_with_labels['Cluster'] = labels

# Feature comparison
feature_to_plot = st.selectbox(
    "Select feature to compare across clusters",
    options=df.columns.tolist()
)

# Only plot clusters (not noise)
df_clusters_only = df_with_labels[df_with_labels['Cluster'] != -1]

if len(df_clusters_only) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    df_clusters_only.boxplot(column=feature_to_plot, by='Cluster', ax=ax)
    ax.set_title(f"{feature_to_plot} Distribution by Cluster (excluding noise)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(feature_to_plot)
    plt.suptitle("")  # Remove the automatic title
    st.pyplot(fig)
    plt.close()
else:
    st.warning("No clusters found with current parameters")

# Show cluster means
st.subheader("Cluster Feature Means")
if n_clusters > 0:
    cluster_means = df_with_labels[df_with_labels['Cluster'] != -1].groupby('Cluster').mean()
    if len(cluster_means) > 0:
        st.dataframe(cluster_means.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
else:
    st.warning("No clusters to display")

# Download clustered data
st.markdown("---")
st.subheader("Download Results")

# Prepare download data
download_df = df_full.copy()
download_df['Cluster'] = labels
download_df['PC1'] = X_2d[:, 0]
download_df['PC2'] = X_2d[:, 1]

csv = download_df.to_csv(index=False)
st.download_button(
    label="Download Clustered Data as CSV",
    data=csv,
    file_name=f"dbscan_clustered_data_eps{eps}_minsamples{min_samples}.csv",
    mime="text/csv"
)

# Info section
with st.expander("ℹ️ About DBSCAN Hyperparameters"):
    st.markdown("""
    ### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    
    **Key Hyperparameters:**
    
    - **Epsilon (eps)**: Maximum distance between two samples for them to be considered neighbors
        - Smaller values → More clusters, more noise
        - Larger values → Fewer, larger clusters
        
    - **Minimum Samples (min_samples)**: Minimum number of points required to form a dense region (cluster)
        - Smaller values → More points clustered, less noise
        - Larger values → More noise points, denser clusters
        
    - **Distance Metric**: Method to calculate distance between points
        - **Euclidean**: Standard straight-line distance
        - **Manhattan**: Sum of absolute differences (good for grid-like data)
        - **Cosine**: Measures angle between vectors (good for high-dimensional data)
    
    ### Advantages of DBSCAN:
    - Can find arbitrarily shaped clusters
    - Automatically identifies outliers (noise points)
    - Doesn't require specifying the number of clusters in advance
    
    ### Tips:
    - Start with eps around the average distance to k-nearest neighbors
    - A good rule of thumb: min_samples ≥ dimensionality + 1
    - If all points are noise, increase eps
    - If all points are in one cluster, decrease eps
    """)
