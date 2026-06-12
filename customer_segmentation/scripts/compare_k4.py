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

# Run KMeans K=4 for both
for name, features in [("With active_months", features_with_active), ("Without active_months", features_without_active)]:
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled[features])
    
    df_temp = df_unscaled.copy()
    df_temp['cluster'] = labels
    
    print(f"\n==========================================")
    print(f"MODEL: {name} (K=4)")
    print(f"==========================================")
    
    # Cluster counts
    counts = df_temp['cluster'].value_counts().sort_index()
    print("Cluster Counts:")
    print(counts)
    
    # Feature Means in Original space
    means = df_temp.groupby('cluster')[features].mean()
    print("\nFeature Means (Original Space):")
    print(means.round(3))
    
    # Centroids in scaled space
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\nCentroids (Scaled Space):")
    print(centroids.round(3))
