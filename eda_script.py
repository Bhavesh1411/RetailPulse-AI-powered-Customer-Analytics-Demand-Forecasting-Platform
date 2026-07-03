import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Setup output directory
out_dir = 'eda_output'
os.makedirs(out_dir, exist_ok=True)

# Load dataset
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data/sales_dataset_online_retail.csv')
print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Ensure date is datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

output = {}

# 1. Dataset shape
output['shape'] = df.shape

# 2. Column listing
output['columns'] = list(df.columns)

# 3. Data types
output['dtypes'] = df.dtypes.astype(str).to_dict()

# 4. Missing value analysis
output['missing_values'] = df.isna().sum().to_dict()

# 5. Duplicate record analysis
output['duplicates'] = int(df.duplicated().sum())

# 6. Unique customer count
if 'customer_id' in df.columns:
    output['unique_customers'] = int(df['customer_id'].nunique())
else:
    output['unique_customers'] = 0

# 7. Unique product count
if 'product_id' in df.columns:
    output['unique_products'] = int(df['product_id'].nunique())
else:
    output['unique_products'] = 0

# 8. Date range analysis
if 'date' in df.columns and not df['date'].isna().all():
    output['date_range'] = {
        'start': str(df['date'].min()),
        'end': str(df['date'].min())
    }
else:
    output['date_range'] = None

# 9. Numerical feature summary
num_cols = df.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    output['numerical_summary'] = df[num_cols].describe().to_dict()
else:
    output['numerical_summary'] = {}

# 10. Categorical feature summary
cat_cols = df.select_dtypes(exclude=[np.number, 'datetime64[ns]']).columns
if len(cat_cols) > 0:
    output['categorical_summary'] = df[cat_cols].describe().to_dict()
else:
    output['categorical_summary'] = {}

# 11. Correlation analysis
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    output['correlation'] = corr.to_dict()
    
    # Visualization: Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'))
    plt.close()

# Visualization: Missing value chart
plt.figure(figsize=(10, 6))
missing_pct = (df.isna().sum() / len(df)) * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
if not missing_pct.empty:
    sns.barplot(x=missing_pct.values, y=missing_pct.index)
    plt.title('Percentage of Missing Values per Column')
    plt.xlabel('Percentage (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'missing_values_chart.png'))
plt.close()

# Visualization: Distribution plots for important numerical columns
for col in ['sales_amount', 'quantity_sold', 'unit_price']:
    if col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=50)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'distribution_{col}.png'))
        plt.close()

# Visualization: Sales trends over time
if 'date' in df.columns and 'sales_amount' in df.columns:
    plt.figure(figsize=(12, 6))
    monthly_sales = df.set_index('date').resample('ME')['sales_amount'].sum()
    monthly_sales.plot()
    plt.title('Monthly Sales Trend')
    plt.ylabel('Total Sales Amount')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sales_trend.png'))
    plt.close()

# Visualization: Top products summary
if 'product_id' in df.columns and 'sales_amount' in df.columns:
    plt.figure(figsize=(10, 6))
    top_products = df.groupby('product_id')['sales_amount'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title('Top 10 Products by Sales Amount')
    plt.xlabel('Sales Amount')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'top_products.png'))
    plt.close()

# Output to JSON
with open(os.path.join(out_dir, 'eda_summary.json'), 'w') as f:
    json.dump(output, f, indent=4)

print("EDA completed successfully. Summary and plots saved in 'eda_output' directory.")
