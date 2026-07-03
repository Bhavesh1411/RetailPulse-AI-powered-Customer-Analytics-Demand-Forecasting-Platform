import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:.4f}'.format)

scaled_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'processed_data/clustering_dataset_std_v2.csv')
original_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'processed_data/customer_segments_kmeans_finalone.csv')

df_scaled = pd.read_csv(scaled_path)
df_original = pd.read_csv(original_path)

features = ['recency', 'frequency', 'monetary', 'average_purchase_value', 'customer_tenure', 'weekend_sales_ratio']
X = df_scaled[features].values

print(f"Dataset shape: {X.shape}")

# =====================================================
# PHASE 1: DBSCAN PARAMETER SELECTION
# =====================================================
print("\n" + "="*60)
print("PHASE 1: DBSCAN PARAMETER SELECTION")
print("="*60)

# K-distance analysis: the elbow is around p90-p95 for min_samples=10
# median ~0.54, p90 ~0.96, p95 ~1.13
# This means the natural neighborhood radius is ~0.5-1.0
# We should search eps in [0.4, 1.2] range

print("\n--- K-Distance Analysis (min_samples=10) ---")
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X)
distances, _ = nn.kneighbors(X)
k_distances = np.sort(distances[:, 9])
for p in [50, 75, 80, 85, 90, 92, 95, 97, 99]:
    print(f"  Percentile {p}: {np.percentile(k_distances, p):.4f}")

print("\n--- DBSCAN Grid Search (Fine-Grained) ---")
best_config = None
best_sil = -1
results = []

for ms in [7, 10, 13]:
    for eps in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        dbscan = DBSCAN(eps=eps, min_samples=ms)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        noise_pct = n_noise / len(labels) * 100
        
        sil = -1.0
        if n_clusters >= 2:
            non_noise_mask = labels != -1
            if len(set(labels[non_noise_mask])) >= 2:
                sil = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
        
        results.append({
            'eps': eps, 'min_samples': ms, 'n_clusters': n_clusters,
            'n_noise': n_noise, 'noise_pct': noise_pct, 'silhouette': sil
        })
        
        # Prefer: 2+ clusters, noise < 50%, highest silhouette
        if sil > best_sil and n_clusters >= 2 and noise_pct < 50:
            best_sil = sil
            best_config = {'eps': eps, 'min_samples': ms}

print(f"{'eps':>5} | {'ms':>3} | {'clusters':>8} | {'noise':>6} | {'noise%':>7} | {'silhouette':>10}")
print("-" * 65)
for r in results:
    marker = " <-- BEST" if best_config and r['eps'] == best_config['eps'] and r['min_samples'] == best_config['min_samples'] else ""
    print(f"{r['eps']:>5.1f} | {r['min_samples']:>3} | {r['n_clusters']:>8} | {r['n_noise']:>6} | {r['noise_pct']:>6.1f}% | {r['silhouette']:>10.4f}{marker}")

# If no multi-cluster solution found, use the best noise-discovery config
if best_config is None:
    # Use eps around elbow for outlier detection
    best_config = {'eps': 1.0, 'min_samples': 10}
    print(f"\nNo multi-cluster solution found. Using eps={best_config['eps']}, min_samples={best_config['min_samples']} for outlier detection.")
else:
    print(f"\nSelected parameters: eps={best_config['eps']}, min_samples={best_config['min_samples']}")
    print(f"Best silhouette (non-noise): {best_sil:.4f}")

# =====================================================
# PHASE 2: DBSCAN EXECUTION
# =====================================================
print("\n" + "="*60)
print("PHASE 2: DBSCAN EXECUTION")
print("="*60)

dbscan = DBSCAN(eps=best_config['eps'], min_samples=best_config['min_samples'])
db_labels = dbscan.fit_predict(X)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
noise_pct = n_noise / len(db_labels) * 100

print(f"Number of clusters discovered: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Percentage of noise: {noise_pct:.2f}%")
print(f"\nCluster sizes:")
for label in sorted(set(db_labels)):
    count = (db_labels == label).sum()
    name = "Noise (-1)" if label == -1 else f"DBSCAN Cluster {label}"
    print(f"  {name}: {count} ({count/len(db_labels)*100:.1f}%)")

# Feature means per DBSCAN cluster (original space)
df_original['dbscan_cluster'] = db_labels
print(f"\nFeature Means by DBSCAN Cluster (Original Space):")
print(df_original.groupby('dbscan_cluster')[features].mean().round(2))

# =====================================================
# PHASE 3: COMPARISON WITH KMEANS
# =====================================================
print("\n" + "="*60)
print("PHASE 3: COMPARISON WITH KMEANS")
print("="*60)

print("\nCross-tabulation: KMeans Cluster vs DBSCAN Cluster")
ct = pd.crosstab(df_original['cluster'], df_original['dbscan_cluster'], margins=True)
print(ct)

print("\nKMeans cluster breakdown within each DBSCAN group:")
for db_c in sorted(set(db_labels)):
    mask = db_labels == db_c
    name = "Noise" if db_c == -1 else f"DBSCAN {db_c}"
    km_dist = df_original.loc[mask, 'cluster'].value_counts()
    print(f"\n  {name} ({mask.sum()} customers):")
    for km_c, cnt in km_dist.items():
        print(f"    {km_c}: {cnt} ({cnt/mask.sum()*100:.1f}%)")

# =====================================================
# PHASE 4: OUTLIER/NOISE ANALYSIS
# =====================================================
print("\n" + "="*60)
print("PHASE 4: OUTLIER CUSTOMER ANALYSIS")
print("="*60)

noise_df = df_original[db_labels == -1].copy()
if len(noise_df) > 0:
    print(f"\nNoise Point Count: {len(noise_df)}")
    print(f"\nNoise Points - Feature Summary (Original Space):")
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(noise_df[features].describe().round(2))
    
    print(f"\nNoise Points - KMeans Distribution:")
    print(noise_df['cluster'].value_counts())
    
    print(f"\nTop 20 Noise Points by Monetary Value:")
    top_noise = noise_df.nlargest(20, 'monetary')[['customer_id', 'recency', 'frequency', 'monetary', 'average_purchase_value', 'customer_tenure', 'weekend_sales_ratio', 'cluster']]
    print(top_noise.to_string(index=False))
    
    # Categorize noise
    ultra_high = noise_df[noise_df['monetary'] > 10000]
    high_freq = noise_df[noise_df['frequency'] > 20]
    high_apv = noise_df[noise_df['average_purchase_value'] > 1000]
    low_value_lapsed = noise_df[(noise_df['monetary'] < 100) & (noise_df['recency'] > 300)]
    
    print(f"\nNoise Sub-Categories:")
    print(f"  Ultra-High Spenders (monetary > $10K):   {len(ultra_high)}")
    print(f"  High-Frequency Buyers (freq > 20):       {len(high_freq)}")
    print(f"  Premium Basket Buyers (APV > $1K):        {len(high_apv)}")
    print(f"  Low-Value Lapsed (monetary<$100, rec>300): {len(low_value_lapsed)}")
    
    if len(ultra_high) > 0:
        print(f"\n  Ultra-High Spender Revenue Impact:")
        print(f"    Combined Spend: ${ultra_high['monetary'].sum():,.2f}")
        print(f"    Avg Spend: ${ultra_high['monetary'].mean():,.2f}")
        print(f"    Max Spend: ${ultra_high['monetary'].max():,.2f}")
        print(f"    Avg Frequency: {ultra_high['frequency'].mean():.1f}")
    
    if len(high_freq) > 0:
        print(f"\n  High-Frequency Buyer Profile:")
        print(f"    Combined Spend: ${high_freq['monetary'].sum():,.2f}")
        print(f"    Avg Frequency: {high_freq['frequency'].mean():.1f}")
        print(f"    Max Frequency: {high_freq['frequency'].max()}")
else:
    print("\nNo noise points detected at this eps/min_samples configuration.")

# Also run a stricter DBSCAN for maximum outlier discovery
print("\n\n--- BONUS: Strict DBSCAN (eps=0.5, min_samples=10) for max outlier discovery ---")
dbscan_strict = DBSCAN(eps=0.5, min_samples=10)
strict_labels = dbscan_strict.fit_predict(X)
strict_noise = (strict_labels == -1).sum()
strict_clusters = len(set(strict_labels)) - (1 if -1 in strict_labels else 0)
print(f"Clusters: {strict_clusters}, Noise: {strict_noise} ({strict_noise/len(strict_labels)*100:.1f}%)")

strict_noise_df = df_original[strict_labels == -1].copy()
if len(strict_noise_df) > 0:
    print(f"\nStrict Noise Points - Feature Summary:")
    print(strict_noise_df[features].describe().round(2))
    
    print(f"\nStrict Noise - KMeans Distribution:")
    print(strict_noise_df['cluster'].value_counts())
    
    strict_ultra = strict_noise_df[strict_noise_df['monetary'] > 10000]
    strict_hfreq = strict_noise_df[strict_noise_df['frequency'] > 20]
    strict_hapv = strict_noise_df[strict_noise_df['average_purchase_value'] > 1000]
    
    print(f"\nStrict Noise Sub-Categories:")
    print(f"  Ultra-High Spenders (monetary > $10K):   {len(strict_ultra)}")
    print(f"  High-Frequency Buyers (freq > 20):       {len(strict_hfreq)}")
    print(f"  Premium Basket Buyers (APV > $1K):        {len(strict_hapv)}")
    
    if len(strict_ultra) > 0:
        print(f"\n  Strict Ultra-High Spenders:")
        top_strict = strict_ultra.nlargest(15, 'monetary')[['customer_id', 'recency', 'frequency', 'monetary', 'average_purchase_value', 'cluster']]
        print(top_strict.to_string(index=False))
        print(f"    Combined Spend: ${strict_ultra['monetary'].sum():,.2f}")

# Save
output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'processed_data/customer_segments_dbscan.csv')
df_original.to_csv(output_path, index=False)
print(f"\nSaved DBSCAN assignments to: {output_path}")
print("\nDone.")
