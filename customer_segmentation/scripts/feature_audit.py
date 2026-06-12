import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"c:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customers_features_finalone.csv")

# Print columns and basic info
print("Dataset columns:")
print(df.columns.tolist())
print("\nShape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Drop non-numeric columns for correlation matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns:", numeric_cols)

# Compute correlation matrix
corr_matrix = df[numeric_cols].corr()

# Find highly correlated pairs
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.5:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

print("\nHighly correlated pairs (|corr| > 0.5):")
for f1, f2, val in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
    print(f"{f1} vs {f2}: {val:.4f}")

# Display description statistics
print("\nSummary stats:")
print(df.describe().T)

# Save correlation matrix to CSV
corr_matrix.to_csv(r"c:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\correlation_matrix.csv")
print("\nSaved correlation matrix to correlation_matrix.csv")
