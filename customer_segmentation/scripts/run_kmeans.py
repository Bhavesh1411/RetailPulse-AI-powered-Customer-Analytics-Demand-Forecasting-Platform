import pandas as pd
from sklearn.cluster import KMeans

# Paths
unscaled_path = r"C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customers_features_finalone.csv"
scaled_path = r"C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\clustering_dataset_std.csv"
output_path = r"C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customer_segments_kmeans.csv"

# Load
df_unscaled = pd.read_csv(unscaled_path)
df_scaled = pd.read_csv(scaled_path)

# Features to use (Without active_months as selected best model)
features = ['recency', 'frequency', 'monetary', 'average_purchase_value', 'customer_tenure', 'weekend_sales_ratio']

# Run KMeans K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(df_scaled[features])

# Format cluster names as "Cluster 0", "Cluster 1", etc.
cluster_names = [f"Cluster {l}" for l in labels]

# Merge cluster labels with original unscaled features
df_output = df_unscaled.copy()
df_output['cluster'] = cluster_names

# Save to CSV
df_output.to_csv(output_path, index=False)
print(f"Successfully generated cluster assignments and saved to {output_path}")
print("Sample output:")
print(df_output[['customer_id', 'cluster']].head())
