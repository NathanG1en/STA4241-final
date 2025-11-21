import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Hierarchical Clustering Explorer", layout="wide")

# Title
st.title("🔍 Hierarchical Clustering Hyperparameter Explorer")
st.markdown("Explore how different hyperparameters affect hierarchical clustering on the Wholesale Customers dataset")

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

# Linkage method
linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    options=["ward", "single", "complete", "average"],
    help="Method to calculate distance between clusters"
)


# Number of clusters
n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=3,
    help="Number of clusters to form"
)

# PCA components (for exploration)
show_pca_variance = st.sidebar.checkbox("Show PCA Explained Variance", value=False)

# Dendrogram options
show_dendrogram = st.sidebar.checkbox("Show Dendrogram", value=True)
dendrogram_truncate = st.sidebar.slider(
    "Dendrogram Truncation Level",
    min_value=0,
    max_value=30,
    value=0,
    help="0 = show all levels, higher = more truncation"
)

# Main content area
col1, col2 = st.columns(2)

# Perform clustering
Z = linkage(X, method=linkage_method)
labels = fcluster(Z, t=n_clusters, criterion='maxclust')

# Compute dendrogram color threshold corresponding to n_clusters
# This finds the height at which cutting produces exactly n_clusters
dists = Z[:, 2]
sorted_dists = np.sort(dists)
if n_clusters - 1 < len(sorted_dists):
    color_threshold = sorted_dists[-n_clusters]
else:
    color_threshold = 0


# Column 1: Clustering visualization
with col1:
    st.subheader("Clustering Results")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=50)
    ax.set_title(f"Hierarchical Clustering ({linkage_method} linkage, k={n_clusters})")
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)
    plt.close()
    
    # Cluster statistics
    st.subheader("Cluster Statistics")
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_df = pd.DataFrame({
        'Cluster': unique_labels,
        'Count': counts,
        'Percentage': (counts / len(labels) * 100).round(2)
    })
    st.dataframe(cluster_df, use_container_width=True)

# Column 2: Additional visualizations
with col2:
    # Dendrogram
    if show_dendrogram:
        st.subheader("Dendrogram")

        fig, ax = plt.subplots(figsize=(8, 6))

        dendrogram(
            Z,
            ax=ax,
            truncate_mode='lastp' if dendrogram_truncate > 0 else None,
            p=dendrogram_truncate if dendrogram_truncate > 0 else None,
            color_threshold=color_threshold
        )

        ax.set_title(f"Dendrogram ({linkage_method} linkage)")
        ax.set_xlabel("Sample Index (or Cluster Size)")
        ax.set_ylabel("Distance")
        st.pyplot(fig)
        plt.close()

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

fig, ax = plt.subplots(figsize=(10, 5))
df_with_labels.boxplot(column=feature_to_plot, by='Cluster', ax=ax)
ax.set_title(f"{feature_to_plot} Distribution by Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel(feature_to_plot)
plt.suptitle("")  # Remove the automatic title
st.pyplot(fig)
plt.close()

# Show cluster means
st.subheader("Cluster Feature Means")
cluster_means = df_with_labels.groupby('Cluster').mean()
st.dataframe(cluster_means.style.background_gradient(cmap='YlOrRd'), use_container_width=True)

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
    file_name=f"clustered_data_{linkage_method}_{n_clusters}clusters.csv",
    mime="text/csv"
)

# Info section
with st.expander("ℹ️ About the Hyperparameters"):
    st.markdown("""
    ### Linkage Methods:
    - **Single**: Minimum distance between any two points in different clusters (prone to chaining)
    - **Complete**: Maximum distance between any two points in different clusters (compact clusters)
    - **Average**: Average distance between all pairs of points in different clusters
    - **Ward**: Minimizes within-cluster variance (tends to produce equal-sized clusters)
    
    ### Number of Clusters:
    - Determines how many groups to segment the data into
    - Use the dendrogram to help decide on an appropriate number
    - Look for large jumps in distance (height) in the dendrogram
    """)
