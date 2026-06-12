import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

# Paths
input_file = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customers_features_finalone.csv'
output_dir = r'C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data'

print("Loading data...")
df = pd.read_csv(input_file)

# Approved features + active_months + customer_id
features = [
    'customer_id', 'recency', 'frequency', 'monetary', 
    'average_purchase_value', 'customer_tenure', 
    'weekend_sales_ratio', 'active_months'
]
df = df[features].copy()

# 1 & 2. Handle nulls and infinites
print(f"Original shape: {df.shape}")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
null_counts = df.isnull().sum()
print("Null values before dropping:")
print(null_counts)
df.dropna(inplace=True)
print(f"Shape after dropping nulls/infs: {df.shape}")

# 3 & 4. Analyze feature distributions and identify skewness
numeric_features = [f for f in features if f != 'customer_id']

print("\n--- Feature Distribution Summary (Before Transform) ---")
skewness_report = []
for col in numeric_features:
    col_skew = skew(df[col])
    print(f"{col}: Skewness = {col_skew:.4f}")
    skewness_report.append({'Feature': col, 'Skewness': col_skew})

# 5. Apply log transformation where statistically justified
# A general rule of thumb is that if skewness is > 1 or < -1, it's highly skewed.
print("\n--- Log Transformation Report ---")
df_transformed = df.copy()
transformed_cols = []
for item in skewness_report:
    col = item['Feature']
    s = item['Skewness']
    if abs(s) > 1.0:
        print(f"Applying log1p transformation to {col} (Skewness: {s:.4f} > 1.0)")
        # using log1p to handle 0s
        # make sure no negative values exist
        min_val = df_transformed[col].min()
        if min_val < 0:
            df_transformed[col] = df_transformed[col] - min_val + 1
        df_transformed[col] = np.log1p(df_transformed[col])
        transformed_cols.append(col)
        new_skew = skew(df_transformed[col])
        print(f"  -> New Skewness: {new_skew:.4f}")
    else:
        print(f"No transformation needed for {col} (Skewness: {s:.4f})")

# 6. Compare StandardScaler vs RobustScaler
print("\n--- Scaling Comparison ---")
# Create separate datasets for comparison
scaler_standard = StandardScaler()
scaler_robust = RobustScaler()

df_std = df_transformed.copy()
df_rob = df_transformed.copy()

df_std[numeric_features] = scaler_standard.fit_transform(df_transformed[numeric_features])
df_rob[numeric_features] = scaler_robust.fit_transform(df_transformed[numeric_features])

# Print summary of scaled datasets to compare
print("\nStandardScaler Summary (mean, std):")
for col in numeric_features:
    print(f"{col} - Mean: {df_std[col].mean():.4f}, Std: {df_std[col].std():.4f}")

print("\nRobustScaler Summary (median, IQR):")
for col in numeric_features:
    q75, q25 = np.percentile(df_rob[col], [75 ,25])
    iqr = q75 - q25
    print(f"{col} - Median: {np.median(df_rob[col]):.4f}, IQR: {iqr:.4f}")
    
# We will save both to compare and let the agent decide
output_file_std = os.path.join(output_dir, 'clustering_dataset_std.csv')
output_file_rob = os.path.join(output_dir, 'clustering_dataset_rob.csv')

df_std.to_csv(output_file_std, index=False)
df_rob.to_csv(output_file_rob, index=False)

print(f"\nSaved Standard Scaled dataset to {output_file_std}")
print(f"Saved Robust Scaled dataset to {output_file_rob}")
print("Done.")
