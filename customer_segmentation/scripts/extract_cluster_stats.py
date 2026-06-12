import pandas as pd
import numpy as np

# The v2 file was saved as the full unscaled dataset with cluster label
# The previous script saved to customer_segments_kmeans_v2.csv but it was
# created from df_clean which was df_original filtered. Let's use the no_refunds file
# plus re-assign clusters from clustering_dataset_std_v2.
# Actually the script saved to customer_segments_kmeans_v2.csv via df_clean.to_csv
# The file may have been saved without the suffix correctly. Let's try the alternative name:
segments_file = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customer_segments_kmeans_.csv'
df = pd.read_csv(segments_file)

features = ['recency', 'frequency', 'monetary', 'average_purchase_value', 'customer_tenure', 'weekend_sales_ratio']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

total = len(df)
print(f"Total Customers: {total}")

counts = df['cluster'].value_counts().sort_index()
print("\nCluster Counts:")
print(counts)

means = df.groupby('cluster')[features].mean()
print("\nFeature Means:")
print(means.round(4))

# Percentages
for c, cnt in counts.items():
    print(f"{c}: {cnt} customers ({cnt/total*100:.1f}%)")

# More stats
for col in features:
    print(f"\n--- {col} ---")
    print(df.groupby('cluster')[col].agg(['min','max','mean','median','std']).round(2))
