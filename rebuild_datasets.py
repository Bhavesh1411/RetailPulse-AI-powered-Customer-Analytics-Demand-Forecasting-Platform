"""
rebuild_datasets.py
RetailPulse: Complete rebuild of Sales & Forecasting Datasets (2009-2010)

Author: RetailPulse AI
Date: 2026-06-11
"""

import pandas as pd
import numpy as np
import os
import json

def main():
    print("=" * 60)
    print("REBUILDING SALES & FORECASTING DATASETS")
    print("=" * 60)

    # Paths
    base_dir = r"C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data"
    input_file = os.path.join(base_dir, "sales_dataset_online_retail.csv")
    cleaned_file = os.path.join(base_dir, "cleaned_sales_dataset.csv")
    enhanced_file = os.path.join(base_dir, "cleaned_sales_dataset_enhanced.csv")
    forecast_file = os.path.join(base_dir, "daily_sales_forecast_features.csv")
    report_file = os.path.join(base_dir, "rebuild_report.json")

    # 1. LOAD SOURCE DATA
    print("Loading source data...")
    df = pd.read_csv(input_file)
    source_rows = len(df)
    
    # Check date range
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min()
    max_date = df['date'].max()
    print(f"Source Rows: {source_rows:,}")
    print(f"Source Date Range: {min_date.date()} to {max_date.date()}")

    # 2. DATA QUALITY CLEANING
    print("\nApplying Data Quality Rules...")
    stats = {
        'source_rows': source_rows,
        'duplicates_removed': 0,
        'zero_price_removed': 0,
        'bad_debt_removed': 0,
        'test_admin_removed': 0,
        'missing_id_handled': 0
    }

    # a. Exact duplicates
    dup_mask = df.duplicated()
    stats['duplicates_removed'] = int(dup_mask.sum())
    df = df[~dup_mask]

    # b. Strip strings
    string_cols = ['transaction_id', 'product_id', 'region']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # c. Handle missing Customer IDs gracefully (do not drop, label as 'GUEST')
    # Actually, in the original pipeline, missing customer_ids were cast to string 'nan' and kept.
    # We will keep them as np.nan or string.
    if 'customer_id' in df.columns:
        df['customer_id'] = df['customer_id'].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', np.nan)
        stats['missing_id_handled'] = int(df['customer_id'].isna().sum())

    # d. Zero/Negative Unit Price (Accounting entries, not sales)
    zero_price_mask = df['unit_price'] <= 0
    stats['zero_price_removed'] = int(zero_price_mask.sum())
    df = df[~zero_price_mask]

    # e. Bad Debt & Test Transactions
    bad_debt_mask = (df['product_id'] == 'B')
    stats['bad_debt_removed'] = int(bad_debt_mask.sum())
    df = df[~bad_debt_mask]

    test_mask = df['product_id'].isin(['TEST001', 'BANK CHARGES'])
    stats['test_admin_removed'] = int(test_mask.sum())
    df = df[~test_mask]

    # f. Validating Returns
    returns_count = df['transaction_id'].str.startswith('C').sum()
    stats['returns_preserved'] = int(returns_count)

    # Empty columns drop
    empty_cols = ['store_id', 'category', 'discount', 'payment_method']
    df.drop(columns=[c for c in empty_cols if c in df.columns], inplace=True, errors='ignore')

    stats['final_cleaned_rows'] = len(df)
    print(f"Cleaned Rows Retained: {stats['final_cleaned_rows']:,} ({(stats['final_cleaned_rows']/source_rows)*100:.1f}%)")

    # Save cleaned sales dataset
    df.to_csv(cleaned_file, index=False)
    print(f"Saved: {cleaned_file}")

    # 3. ENHANCED SALES DATASET
    print("\nGenerating Enhanced Features...")
    df_enh = df.copy()
    df_enh['year'] = df_enh['date'].dt.year
    df_enh['month'] = df_enh['date'].dt.month
    df_enh['quarter'] = df_enh['date'].dt.quarter
    df_enh['week_of_year'] = df_enh['date'].dt.isocalendar().week.astype(int)
    df_enh['day_of_week'] = df_enh['date'].dt.dayofweek
    df_enh['day_name'] = df_enh['date'].dt.day_name()
    df_enh['is_weekend'] = df_enh['day_of_week'].isin([5, 6]).astype(int)
    df_enh['sales_per_unit'] = np.where(df_enh['quantity_sold'] != 0, df_enh['sales_amount'] / df_enh['quantity_sold'], 0.0)

    df_enh.to_csv(enhanced_file, index=False)
    print(f"Saved: {enhanced_file}")

    # 4. FORECASTING DATASET (Daily Aggregation)
    print("\nGenerating Daily Sales Forecasting Features...")
    df_enh['day_date'] = df_enh['date'].dt.normalize()
    daily_df = df_enh.groupby('day_date').agg(
        sales_amount=('sales_amount', 'sum'),
        quantity_sold=('quantity_sold', 'sum')
    ).reset_index()

    # Reindex to fill missing days with 0
    daily_df.set_index('day_date', inplace=True)
    full_date_range = pd.date_range(start=daily_df.index.min(), end=daily_df.index.max(), freq='D')
    daily_df = daily_df.reindex(full_date_range, fill_value=0.0).reset_index()
    daily_df.rename(columns={'index': 'date'}, inplace=True)

    # Feature Engineering
    overall_mean_sales = daily_df['sales_amount'].mean()

    # Rolling means
    daily_df['rolling_7_day_sales'] = daily_df['sales_amount'].rolling(window=7, min_periods=1).mean()
    daily_df['rolling_30_day_sales'] = daily_df['sales_amount'].rolling(window=30, min_periods=1).mean()
    daily_df['rolling_7_day_quantity'] = daily_df['quantity_sold'].rolling(window=7, min_periods=1).mean()
    daily_df['rolling_30_day_quantity'] = daily_df['quantity_sold'].rolling(window=30, min_periods=1).mean()

    # Lags
    for lag in [1, 7, 30]:
        daily_df[f'lag_{lag}_sales'] = daily_df['sales_amount'].shift(lag)
        daily_df[f'lag_{lag}_qty'] = daily_df['quantity_sold'].shift(lag)

    # Seasonality
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    wd_mean = daily_df.groupby('day_of_week')['sales_amount'].mean().reset_index(name='wd_mean')
    wd_mean['weekday_seasonality_index'] = wd_mean['wd_mean'] / (overall_mean_sales + 1e-5)
    daily_df = daily_df.merge(wd_mean[['day_of_week', 'weekday_seasonality_index']], on='day_of_week', how='left')

    daily_df['month'] = daily_df['date'].dt.month
    mo_mean = daily_df.groupby('month')['sales_amount'].mean().reset_index(name='mo_mean')
    mo_mean['monthly_seasonality_index'] = mo_mean['mo_mean'] / (overall_mean_sales + 1e-5)
    daily_df = daily_df.merge(mo_mean[['month', 'monthly_seasonality_index']], on='month', how='left')

    daily_df['seasonality_index'] = daily_df['weekday_seasonality_index'] * daily_df['monthly_seasonality_index']
    daily_df.drop(columns=['day_of_week', 'month'], inplace=True)

    # MoM Growth (Lagged to prevent leakage)
    daily_df['year_month'] = daily_df['date'].dt.to_period('M')
    monthly_sales = daily_df.groupby('year_month')['sales_amount'].sum().reset_index(name='monthly_sales')
    # Shift by 1 to get previous month's sales, then calculate pct change to the month before that
    # Basically: Growth available in month M is growth of (M-1) over (M-2)
    monthly_sales['prev_month_sales'] = monthly_sales['monthly_sales'].shift(1)
    monthly_sales['prev_prev_month_sales'] = monthly_sales['monthly_sales'].shift(2)
    monthly_sales['sales_growth_rate'] = np.where(
        monthly_sales['prev_prev_month_sales'] > 0,
        (monthly_sales['prev_month_sales'] - monthly_sales['prev_prev_month_sales']) / monthly_sales['prev_prev_month_sales'],
        0.0
    )
    daily_df = daily_df.merge(monthly_sales[['year_month', 'sales_growth_rate']], on='year_month', how='left')
    daily_df.drop(columns=['year_month'], inplace=True)

    # Volatility & Momentum
    daily_df['sales_volatility_30_day'] = daily_df['sales_amount'].rolling(window=30, min_periods=1).std().fillna(0.0)
    daily_df['demand_momentum'] = np.where(
        daily_df['rolling_30_day_sales'] > 0,
        daily_df['rolling_7_day_sales'] / daily_df['rolling_30_day_sales'],
        1.0
    )
    daily_df['quantity_momentum'] = np.where(
        daily_df['rolling_30_day_quantity'] > 0,
        daily_df['rolling_7_day_quantity'] / daily_df['rolling_30_day_quantity'],
        1.0
    )

    daily_df.to_csv(forecast_file, index=False)
    print(f"Saved: {forecast_file}")

    # 5. REPORT GENERATION
    print("\nGenerating Report...")
    report = {
        "dataset_consistency": {
            "source_data": {
                "file": "sales_dataset_online_retail.csv",
                "rows": stats['source_rows'],
                "date_range": f"{min_date.date()} to {max_date.date()}"
            },
            "cleaned_sales_dataset": {
                "file": "cleaned_sales_dataset.csv",
                "rows": stats['final_cleaned_rows'],
                "date_range": f"{min_date.date()} to {max_date.date()}"
            },
            "cleaned_sales_dataset_enhanced": {
                "file": "cleaned_sales_dataset_enhanced.csv",
                "rows": stats['final_cleaned_rows'],
                "date_range": f"{min_date.date()} to {max_date.date()}"
            },
            "daily_sales_forecast_features": {
                "file": "daily_sales_forecast_features.csv",
                "rows": len(daily_df),
                "date_range": f"{min_date.date()} to {max_date.date()}"
            }
        },
        "data_quality_removals": {
            "total_removed": stats['source_rows'] - stats['final_cleaned_rows'],
            "exact_duplicates": stats['duplicates_removed'],
            "zero_or_negative_price": stats['zero_price_removed'],
            "bad_debt_adjustments": stats['bad_debt_removed'],
            "test_and_admin_transactions": stats['test_admin_removed']
        },
        "data_quality_retained": {
            "total_retained": stats['final_cleaned_rows'],
            "returns_preserved_count": stats['returns_preserved'],
            "retention_percentage": f"{(stats['final_cleaned_rows']/stats['source_rows'])*100:.1f}%"
        }
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Report written to {report_file}")
    print("=" * 60)
    print("REBUILD COMPLETE")

if __name__ == "__main__":
    main()
