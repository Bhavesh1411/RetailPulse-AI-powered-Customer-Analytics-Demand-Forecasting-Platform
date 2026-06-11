"""
build_product_forecast_dataset.py
RetailPulse: Generating a granular Product-Level Daily Forecasting Dataset

Author: RetailPulse AI
Date: 2026-06-11
"""

import pandas as pd
import numpy as np
import os

def main():
    print("=" * 60)
    print("BUILDING PRODUCT-LEVEL DAILY FORECASTING DATASET")
    print("=" * 60)

    # Paths
    base_dir = r"C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data"
    input_file = os.path.join(base_dir, "cleaned_sales_dataset.csv")
    output_file = os.path.join(base_dir, "product_daily_forecast_features.csv")

    print(f"Loading cleaned dataset: {input_file} ...")
    df = pd.read_csv(input_file)
    df['day_date'] = pd.to_datetime(df['date']).dt.normalize()

    # Aggregate by day and product
    print("Aggregating by Date and Product ID...")
    daily_product = df.groupby(['day_date', 'product_id']).agg(
        daily_sales_amount=('sales_amount', 'sum'),
        daily_quantity_sold=('quantity_sold', 'sum')
    ).reset_index()

    # Sort to ensure chronological order per product
    daily_product.sort_values(by=['product_id', 'day_date'], inplace=True)
    daily_product.reset_index(drop=True, inplace=True)

    print(f"Base Aggregation Shape: {daily_product.shape}")

    # Feature Engineering (Grouped by Product)
    print("Calculating Rolling Features and Lags per Product...")
    
    # We use expanding/rolling within groups. 
    # Because days might be sparse (not all products sell every day), a rolling window of 7 rows 
    # means the last 7 *transactions* for that product, rather than exactly 7 calendar days. 
    # For many ML models, this sequence-based lag is highly effective.
    # If we wanted strict calendar days, we would have to reindex every product against every calendar day,
    # resulting in 374 days * 4262 products = 1,593,988 rows (mostly zeros). 
    # Given the 150k-300k target, the sparse chronological sequence is preferred.

    # Lags
    daily_product['product_lag_1_sales'] = daily_product.groupby('product_id')['daily_sales_amount'].shift(1)
    daily_product['product_lag_7_sales'] = daily_product.groupby('product_id')['daily_sales_amount'].shift(7)
    daily_product['product_lag_1_qty'] = daily_product.groupby('product_id')['daily_quantity_sold'].shift(1)
    daily_product['product_lag_7_qty'] = daily_product.groupby('product_id')['daily_quantity_sold'].shift(7)

    # Rolling Means
    daily_product['product_rolling_7_sales'] = daily_product.groupby('product_id')['daily_sales_amount'].rolling(window=7, min_periods=1).mean().reset_index(drop=True)
    daily_product['product_rolling_30_sales'] = daily_product.groupby('product_id')['daily_sales_amount'].rolling(window=30, min_periods=1).mean().reset_index(drop=True)
    daily_product['product_rolling_7_qty'] = daily_product.groupby('product_id')['daily_quantity_sold'].rolling(window=7, min_periods=1).mean().reset_index(drop=True)
    daily_product['product_rolling_30_qty'] = daily_product.groupby('product_id')['daily_quantity_sold'].rolling(window=30, min_periods=1).mean().reset_index(drop=True)

    # Demand Momentum
    daily_product['product_demand_momentum'] = np.where(
        daily_product['product_rolling_30_sales'] > 0,
        daily_product['product_rolling_7_sales'] / daily_product['product_rolling_30_sales'],
        1.0
    )

    # Global Seasonality Injection
    # We can inject the global day-of-week and month index so the product models know the global trend
    print("Injecting Global Seasonality Indices...")
    daily_product['day_of_week'] = daily_product['day_date'].dt.dayofweek
    daily_product['month'] = daily_product['day_date'].dt.month

    # Calculate global means
    global_daily = df.groupby('day_date')['sales_amount'].sum().reset_index()
    global_mean = global_daily['sales_amount'].mean()

    # Day of week global index
    dow_df = df.copy()
    dow_df['dow'] = dow_df['day_date'].dt.dayofweek
    dow_mean = dow_df.groupby('dow')['sales_amount'].sum() / dow_df.groupby('dow')['day_date'].nunique()
    wd_index = (dow_mean / global_mean).reset_index()
    wd_index.columns = ['day_of_week', 'global_weekday_seasonality']
    
    # Month global index
    mo_df = df.copy()
    mo_df['mo'] = mo_df['day_date'].dt.month
    mo_mean = mo_df.groupby('mo')['sales_amount'].sum() / mo_df.groupby('mo')['day_date'].nunique()
    mo_index = (mo_mean / global_mean).reset_index()
    mo_index.columns = ['month', 'global_monthly_seasonality']

    daily_product = daily_product.merge(wd_index, on='day_of_week', how='left')
    daily_product = daily_product.merge(mo_index, on='month', how='left')
    daily_product.drop(columns=['day_of_week', 'month'], inplace=True)

    # Clean and order columns
    daily_product.rename(columns={'day_date': 'date'}, inplace=True)
    
    # Save output
    print(f"\nFinal Shape: {daily_product.shape}")
    print(f"Saving to {output_file} ...")
    daily_product.to_csv(output_file, index=False)
    
    print("=" * 60)
    print("SUCCESS")
    print(f"Product-Level Forecast Dataset built with {len(daily_product):,} rows!")
    print("=" * 60)

if __name__ == "__main__":
    main()
