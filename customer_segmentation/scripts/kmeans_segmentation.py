import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load datasets
unscaled_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'processed_data/customers_features_finalone.csv')
scaled_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'processed_data/clustering_dataset_std.csv')

df_unscaled = pd.read_csv(unscaled_path)
df_scaled = pd.read_csv(scaled_path)

# Verify customer ids match
assert (df_unscaled['customer_id'].astype(str) == df_scaled['customer_id'].astype(str)).all(), "Customer IDs do not match!"

# Define feature sets
features_with_active = [
    'recency', 'frequency', 'monetary', 'average_purchase_value', 
    'customer_tenure', 'weekend_sales_ratio', 'active_months'
]
features_without_active = [
    'recency', 'frequency', 'monetary', 'average_purchase_value', 
    'customer_tenure', 'weekend_sales_ratio'
]

datasets = {
    "With active_months": df_scaled[features_with_active],
    "Without active_months": df_scaled[features_without_active]
}

# Run metrics for K from 2 to 8
results = {}

for name, X in datasets.items():
    print(f"\nEvaluating KMeans for: {name}")
    results[name] = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss = kmeans.inertia_
        sil = silhouette_score(X, labels, random_state=42)
        db = davies_bouldin_score(X, labels)
        results[name].append({
            'k': k,
            'wcss': wcss,
            'silhouette': sil,
            'davies_bouldin': db
        })
        print(f"  k={k} | WCSS={wcss:.2f} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

# Print metric summary
for name in datasets.keys():
    print(f"\nSummary metrics for {name}:")
    for res in results[name]:
        print(f"  k={res['k']}: WCSS={res['wcss']:.2f}, Sil={res['silhouette']:.4f}, DB={res['davies_bouldin']:.4f}")

# Let's perform KMeans with K=3, 4, 5 for both and inspect centroids/means to make an informed recommendation.
# We will output cluster counts and centroids for comparison.
for name, X in datasets.items():
    print(f"\n--- Model details for {name} ---")
    for k in [3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Cluster counts
        counts = pd.Series(labels).value_counts().sort_index()
        print(f"\nCluster counts for k={k}:")
        print(counts.to_dict())
        
        # Scaled centroids (centroids in scaled space)
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
        print(f"Scaled Centroids for k={k}:")
        print(centroids.round(3))
        
        # Unscaled feature means by cluster
        df_temp = df_unscaled.copy()
        df_temp['cluster'] = labels
        feature_cols = X.columns.tolist()
        means = df_temp.groupby('cluster')[feature_cols].mean()
        print(f"Original Feature Means for k={k}:")
        print(means.round(3))

# We will write a function to save the final assignment once we inspect the results and select the best model.
# But since the script runs first, let's also automatically output the segment assignments for a few candidate models
# so we have them, or we can choose the best one in the script if we can analyze it, or we can run this script to get the metrics,
# then run another script or update it to write the final file.
# Actually, let's write out the results for both with and without active_months for k=4, which is a very common K, 
# and we can run a final model selection after.
# Wait, let's save the model assignment for K=4 (Without active_months) to customer_segments_kmeans.csv as a placeholder or final depending on analysis.
# Let's look at the metrics first. We will write the script to run and output all evaluation metrics, cluster centroids and counts to stdout.
