import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import json
import os
import warnings
warnings.filterwarnings("ignore")

def perform_adf_test(series):
    result = adfuller(series.dropna())
    return {
        "test_statistic": float(result[0]),
        "p_value": float(result[1]),
        "is_stationary": bool(result[1] < 0.05)
    }

def audit_dataset(df, date_col, target_col, is_panel=False, group_col=None):
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Missing values
    missing_vals = df.isnull().sum().to_dict()
    
    # Duplicates
    duplicates = int(df.duplicated().sum())
    
    # Date ranges
    min_date = str(df[date_col].min().date())
    max_date = str(df[date_col].max().date())
    
    # Chronological ordering
    if not is_panel:
        is_sorted = bool(df[date_col].is_monotonic_increasing)
        
        # Missing dates
        full_date_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq='D')
        missing_dates = int(len(full_date_range) - df[date_col].nunique())
    else:
        # Check for one product as example
        sample_group = df[df[group_col] == df[group_col].iloc[0]]
        is_sorted = bool(sample_group[date_col].is_monotonic_increasing)
        full_date_range = pd.date_range(start=sample_group[date_col].min(), end=sample_group[date_col].max(), freq='D')
        missing_dates = int(len(full_date_range) - sample_group[date_col].nunique())

    # Outliers
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = int(((df[target_col] < lower_bound) | (df[target_col] > upper_bound)).sum())

    return {
        "row_count": len(df),
        "min_date": min_date,
        "max_date": max_date,
        "missing_values": {k: int(v) for k, v in missing_vals.items() if v > 0},
        "duplicates": duplicates,
        "is_sorted": is_sorted,
        "missing_dates": missing_dates,
        "outliers_count": outliers,
        "outliers_pct": round(outliers / len(df) * 100, 2)
    }

def main():
    base_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets')
    daily_path = os.path.join(base_path, "daily_sales_forecast_features.csv")
    product_path = os.path.join(base_path, "product_daily_forecast_features.csv")

    results = {}

    # Load data
    df_daily = pd.read_csv(daily_path)
    df_product = pd.read_csv(product_path)

    # 1. Dataset Comparison & Audit
    results['daily_sales'] = audit_dataset(df_daily, date_col='date', target_col='sales_amount')
    results['product_sales'] = audit_dataset(df_product, date_col='date', target_col='daily_quantity_sold', is_panel=True, group_col='product_id')
    
    results['daily_sales']['columns'] = list(df_daily.columns)
    results['product_sales']['columns'] = list(df_product.columns)

    # 3. Stationarity
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily.set_index('date', inplace=True)
    df_daily.sort_index(inplace=True)
    
    results['stationarity'] = {}
    results['stationarity']['daily_sales_amount'] = perform_adf_test(df_daily['sales_amount'])
    results['stationarity']['daily_quantity'] = perform_adf_test(df_daily['quantity_sold'])
    
    # 4. Seasonal Decomposition
    # Using statsmodels seasonal_decompose (period=7 for weekly)
    try:
        decomp = seasonal_decompose(df_daily['sales_amount'], model='additive', period=7)
        results['decomposition'] = {
            "trend_mean": float(decomp.trend.mean()),
            "seasonal_amplitude": float(decomp.seasonal.max() - decomp.seasonal.min()),
            "residual_std": float(decomp.resid.std())
        }
    except Exception as e:
        results['decomposition'] = {"error": str(e)}

    # Save to JSON
    out_file = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\ef499792-a9c3-4bd3-a984-6da54fe0fe7d\scratch\forecasting_audit.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
