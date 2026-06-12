import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import skew
import os

# Paths
input_features_file = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customers_features_finalone.csv'
output_cleaned_features = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customers_features_no_refunds.csv'
output_scaled_dataset = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\clustering_dataset_std_v2.csv'
output_segments_file = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customer_segments_kmeans_v2.csv'

df_original = pd.read_csv(input_features_file)
original_count = len(df_original)

# 1. Identify refund-dominated customers
# Criteria: monetary clipped to 0, OR negative/zero average quantities.
refund_mask = (df_original['monetary'] == 0.0) | (df_original['quantity_per_order'] <= 0)
df_refunds = df_original[refund_mask].copy()

removed_count = len(df_refunds)
percentage_removed = (removed_count / original_count) * 100

print(f"--- REFUND AUDIT ---")
print(f"Original Customers: {original_count}")
print(f"Number of customers removed: {removed_count}")
print(f"Percentage of total customers: {percentage_removed:.2f}%")
print(f"Median Monetary of removed: {df_refunds['monetary'].median()}")
print(f"Median Quantity of removed: {df_refunds['quantity_per_order'].median()}")

# 2. Create cleaned dataset
df_clean = df_original[~refund_mask].copy()
new_count = len(df_clean)
print(f"\nNew Customer Count: {new_count}")

# Save the cleaned features dataset without anomalies
df_clean.to_csv(output_cleaned_features, index=False)

# 3. Re-run Clustering Dataset Generation (Log Transform + StandardScaler)
features = [
    'recency', 'frequency', 'monetary', 'average_purchase_value', 
    'customer_tenure', 'weekend_sales_ratio'
]

df_clustering = df_clean[features + ['customer_id']].copy()

# Handle NaNs / Infs (should be none, but just in case)
df_clustering.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clustering.dropna(inplace=True)

# Log Transformations (same rules: skew > 1)
numeric_features = features
for col in numeric_features:
    col_skew = skew(df_clustering[col])
    if abs(col_skew) > 1.0:
        min_val = df_clustering[col].min()
        if min_val < 0:
            df_clustering[col] = df_clustering[col] - min_val + 1
        df_clustering[col] = np.log1p(df_clustering[col])

# StandardScaler
scaler = StandardScaler()
df_scaled = df_clustering.copy()
df_scaled[features] = scaler.fit_transform(df_clustering[features])
df_scaled.to_csv(output_scaled_dataset, index=False)
print(f"New clustering dataset size: {df_scaled.shape}")

# 4. KMeans Evaluation on new dataset
print(f"\n--- KMEANS EVALUATION ---")
results = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled[features])
    wcss = kmeans.inertia_
    sil = silhouette_score(df_scaled[features], labels, random_state=42)
    db = davies_bouldin_score(df_scaled[features], labels)
    results.append({'k': k, 'wcss': wcss, 'silhouette': sil, 'db': db})
    print(f"k={k} | WCSS={wcss:.2f} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

# Find best K based on max silhouette
best_k = max(results, key=lambda x: x['silhouette'])['k']
best_sil = max(results, key=lambda x: x['silhouette'])['silhouette']

print(f"\nNew optimal K (based on Silhouette): {best_k}")
print(f"New silhouette score: {best_sil:.4f}")

# 5. Run Best Model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = kmeans.fit_predict(df_scaled[features])

cluster_names = [f"Cluster {l}" for l in final_labels]

# Output counts and unscaled feature means for the best model
df_clean['cluster'] = cluster_names
counts = df_clean['cluster'].value_counts().sort_index()
print(f"\nNew cluster sizes for K={best_k}:")
print(counts.to_dict())

print(f"\nFeature Means (Original Space) for K={best_k}:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_clean.groupby('cluster')[features].mean().round(2))

# Save the final assignments
df_clean.to_csv(output_segments_file, index=False)
print(f"\nSaved new cluster assignments to {output_segments_file}")

