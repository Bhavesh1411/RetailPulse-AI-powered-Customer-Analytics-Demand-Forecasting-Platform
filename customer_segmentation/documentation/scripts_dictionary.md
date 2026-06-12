# Customer Segmentation Scripts Dictionary

This document details the Python scripts included in the `customer_segmentation/scripts/` folder. These scripts implement the feature analysis, preprocessing, clustering execution, anomaly detection, and validation workflows.

### 1. `feature_audit.py`
* **Purpose**: Inspects raw customer features to identify missing values, infinite entries, skewness, and multi-collinearity.
* **Key Actions**: Imputes missing values, trims infs, calculates kurtosis/skewness, and outputs a correlation matrix.

### 2. `prepare_segmentation_dataset.py`
* **Purpose**: Performs log-transformations on highly skewed features and prepares scaled datasets using different scaling strategies.
* **Key Actions**: Compares `StandardScaler` and `RobustScaler` performance and outputs initial preprocessed matrices.

### 3. `kmeans_segmentation.py`
* **Purpose**: Houses core K-Means logic, containing modular functions to run models, evaluate metrics, and plot diagnostics.
* **Key Actions**: Computes Silhouette scores, Davies-Bouldin indices, and inertia values across multiple values of K.

### 4. `run_kmeans.py`
* **Purpose**: Executable script that runs K-Means on the initial standardized dataset.
* **Key Actions**: Tests clustering solutions with and without the optional `active_months` feature to guide final feature selection.

### 5. `compare_k4.py`
* **Purpose**: Compares KMeans cluster characteristics between $K=3$ and $K=4$ options.
* **Key Actions**: Profiles cluster sizes and centroids to determine the optimal number of segments.

### 6. `dump_means.py`
* **Purpose**: Quick utility script to output feature means grouped by cluster.
* **Key Actions**: Prints a readable summary table of cluster centroids directly to the console.

### 7. `audit_cluster_1.py`
* **Purpose**: Investigates a suspicious, small cluster (Cluster 1) to determine if it represents genuine customers or data/transaction artifacts.
* **Key Actions**: Cross-references Cluster 1 customer IDs with original invoices, uncovering extreme refund/cancellation behavior.

### 8. `remove_refunds_and_recluster.py`
* **Purpose**: Filters out refund-dominated customers, re-scales the features, and fits the finalized K-Means model.
* **Key Actions**: Produces the final clean datasets and final segment labels (`customer_segments_kmeans_finalone.csv`).

### 9. `extract_cluster_stats.py`
* **Purpose**: Profiles the final clusters mathematically to extract business insights and define personas.
* **Key Actions**: Computes cluster averages for all features and prints a structured markdown table.

### 10. `dbscan_analysis.py`
* **Purpose**: Validates the K-Means clusters and flags anomalies using DBSCAN density-based clustering.
* **Key Actions**: Implements nearest-neighbor distance calculation (k-dist plot) to locate optimal parameters, runs DBSCAN, and extracts outlier statistics.
