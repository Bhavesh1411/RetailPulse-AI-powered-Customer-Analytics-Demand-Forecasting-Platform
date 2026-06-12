import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

unscaled_path = r"C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customers_features_finalone.csv"
scaled_path = r"C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\clustering_dataset_std.csv"

df_unscaled = pd.read_csv(unscaled_path)
df_scaled = pd.read_csv(scaled_path)

features_with_active = [
    'recency', 'frequency', 'monetary', 'average_purchase_value', 
    'customer_tenure', 'weekend_sales_ratio', 'active_months'
]
features_without_active = [
    'recency', 'frequency', 'monetary', 'average_purchase_value', 
    'customer_tenure', 'weekend_sales_ratio'
]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

for name, features in [("With active_months", features_with_active), ("Without active_months", features_without_active)]:
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled[features])
    
    df_temp = df_unscaled.copy()
    df_temp['cluster'] = labels
    
    # We want to display original feature means for ALL features in features_with_active/features_without_active
    means = df_temp.groupby('cluster')[features].mean()
    print(f"\nMODEL: {name} (K=4) - Feature Means in Original Space:")
    print(means.round(2))
    
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print(f"\nMODEL: {name} (K=4) - Centroids in Scaled Space:")
    print(centroids.round(3))
