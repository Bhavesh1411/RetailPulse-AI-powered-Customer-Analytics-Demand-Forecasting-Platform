import pandas as pd
import numpy as np

# Paths
kmeans_file = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customer_segments_kmeans.csv'
features_file = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customers_features_finalone.csv'

# Load
df_kmeans = pd.read_csv(kmeans_file)
df_features = pd.read_csv(features_file)

# Extract cluster 1
cluster_1 = df_kmeans[df_kmeans['cluster'] == 'Cluster 1']
cluster_1_ids = cluster_1['customer_id'].tolist()
c1_data = df_features[df_features['customer_id'].isin(cluster_1_ids)]

print("Cluster 1 Count:", len(c1_data))

print("\n--- Compare Cluster 1 against Overall Dataset ---")
cols = ['monetary', 'average_purchase_value', 'frequency', 'recency', 'customer_tenure']
overall_means = df_features[cols].mean()
c1_means = c1_data[cols].mean()

for col in cols:
    print(f"{col}: Overall Mean = {overall_means[col]:.2f} | Cluster 1 Mean = {c1_means[col]:.2f}")

print("\n--- Cluster 1 Specific Checks ---")
print("Min Monetary Value:", c1_data['monetary'].min())
print("Max Monetary Value:", c1_data['monetary'].max())
print("Median Monetary Value:", c1_data['monetary'].median())
print("Count of Negative Monetary:", (c1_data['monetary'] < 0).sum())
print("Count of Monetary < 1.0:", (c1_data['monetary'] < 1.0).sum())
print("Count of Monetary == 0:", (c1_data['monetary'] == 0).sum())

print("\n--- Quantity / Refund Checks ---")
# Using quantity_per_order or total_orders if available
if 'quantity_per_order' in c1_data.columns:
    print("Min Quantity Per Order:", c1_data['quantity_per_order'].min())
    print("Count of Negative Quantity:", (c1_data['quantity_per_order'] < 0).sum())

print("\n--- Top 10 Sample Records from Cluster 1 ---")
# Print the specific columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(c1_data[['customer_id', 'monetary', 'frequency', 'recency', 'average_purchase_value', 'quantity_per_order']].head(10).to_string())

# Check for refund/negative transaction logic in engineer_features.py if possible, or just deduce from values.
