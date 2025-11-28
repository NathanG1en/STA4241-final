
---

# STA4241 Final Project

> **Exploring Hierarchical Clustering and DBSCAN w/ a real dataset!**

For STA4241, this repo is meant to run us through two clustering techniques, Hierarchical Clustering and DBSCAN. We explore them through notebooks that are typical for these types of analyses, and we've also developed two webapps that allow us to further play witih and understand the parameters and outputs in the models. 

---

## 🚀 Features


### 📓 Jupyter Notebooks

* **From-scratch implementations**
* **Comprehensive analysis** with internal and external validation metrics
---

### 🌳 Hierarchical Clustering Explorer

* **Interactive dendrograms** with adjustable truncation
* **Multiple linkage methods**: Ward, Single, Complete, Average
* **Real-time clustering** with adjustable cluster count
* **Feature analysis** across clusters with boxplots and heatmaps
* **PCA visualization** with explained variance analysis

### 🎯 DBSCAN Explorer

* **Density-based clustering** with epsilon and min_samples tuning
* **Automatic outlier detection** with noise point visualization
* **Quality metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz
* **Multiple distance metrics**: Euclidean, Manhattan, Cosine
* **Interactive parameter exploration** to understand density-based clustering



## 📦 Installation

### Prerequisites

* Python 3.11+
* UV package manager (recommended)

### Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/NathanG1en/STA4241-final.git
cd clustering-explorer
```

2. **Install dependencies**

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Launch the apps**

```bash
# Hierarchical Clustering Explorer
streamlit run HierarchicalClusteringExplorer.py

# DBSCAN Explorer
streamlit run DBSCAN_Hyperparameter_Explorer.py
```

---

## 📊 Dataset

The project uses the **UCI Wholesale Customers Dataset**, containing annual spending data across six product categories:

* 🥩 Fresh
* 🥛 Milk
* 🛒 Grocery
* 🧊 Frozen
* 🧼 Detergents_Paper
* 🍰 Delicassen

**440 samples** · **6 features** · **Real-world business data**

---
## Each App

### Hierarchical Clustering Explorer

**What you can do:**

* 🎛️ Adjust linkage method and cluster count
* 📈 Inspect dendrograms
* 🎨 View clusters in PCA-reduced space
* 📊 Compare cluster distributions
* 💾 Export clustered CSVs

**Key Parameters:**

* Linkage Method
* Number of Clusters
* Dendrogram Truncation

---

### DBSCAN Explorer

**What you can do:**

* Tune epsilon + min_samples
* Visualize cluster density and outliers
* Track metrics in real-time
* Switch between distance metrics
* Save labeled output

**Key Parameters:**

* **Epsilon (eps)**: 0.1–3.0
* **Minimum Samples**: 2–20
* **Distance Metric**: Euclidean / Manhattan / Cosine

---

## 📚 Notebooks

### `HCA-scratch.ipynb`

**Hierarchical Clustering from Scratch**

Includes:

* Manual distance functions
* All linkage methods
* Side-by-side comparisons

### `HCA-Practical.ipynb`

**HCA Deep Dive**

Includes:

* Internal validation
* External validation (ARI, NMI)
* Visualizations
* Cluster profiling

---

### `DB-scan-Practical.ipynb`

**DBSCAN Deep Dive**

Includes:

* Internal validation
* External validation (ARI, NMI)
* Visualizations
* Cluster profiling

---

## 🎨 Visualizations

The project includes the following graphics:

* PCA scatter plots
* Dendrograms
* Box plots
* Heatmaps
* Explained variance charts'
---

## 🧮 Metrics & Evaluation

### Internal Metrics

* **Silhouette Score** — higher == better
* **Davies-Bouldin Index** — lower == better
* **Calinski-Harabasz Score** — higher == better

### External Metrics (in notebooks)
we were given some labels that could possibly serve as ground truths, so we use the following metrics to see hwo they line up:
* **ARI (Adjusted Rand Index)**
* **NMI (Normalized Mutual Information)**

---

## 🛠️ What We Used

| Category        | Tools                      |
| --------------- | -------------------------- |
| Core            | Python 3.11, NumPy, Pandas |
| ML              | scikit-learn, SciPy        |
| Visualization   | Matplotlib, Seaborn        |
| Web App         | Streamlit                  |
| Notebooks       | Jupyter                    |
| Package Manager | UV                         |

---

## 📖 Learning

### Understanding the Algorithms

**Hierarchical Clustering**

* Agglomerative tree-building
* Dendrogram-based cluster discovery
* Linkage defines cluster shape

**DBSCAN**

* Density-based clustering
* Automatically identifies noise
* No fixed k required

### Tips

1. Start with defaults
2. Use dendrogram for k
3. Check internal metrics
4. Explore freely

---

## Source of data

* UCI Machine Learning Repository
