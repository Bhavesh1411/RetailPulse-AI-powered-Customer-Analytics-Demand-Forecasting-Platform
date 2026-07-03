import pandas as pd
import numpy as np
import os

def main():
    filepath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/daily_sales_forecast_features.csv')
    print(f"Loading dataset from: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    print("\n--- DATASET DETAILS ---")
    print(f"Row count: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Check calendar features
    calendar_cols = ['day_of_week', 'month', 'quarter', 'week_of_year']
    existing_calendar = [c for c in calendar_cols if c in df.columns]
    missing_calendar = [c for c in calendar_cols if c not in df.columns]
    
    print("\n--- CALENDAR FEATURE STATUS ---")
    print(f"Existing calendar features: {existing_calendar}")
    print(f"Missing calendar features: {missing_calendar}")
    
    # Safe Feature Engineering
    if missing_calendar:
        print(f"\nAppending missing calendar features: {missing_calendar}...")
        for col in missing_calendar:
            if col == 'day_of_week':
                df['day_of_week'] = df['date'].dt.dayofweek
            elif col == 'month':
                df['month'] = df['date'].dt.month
            elif col == 'quarter':
                df['quarter'] = df['date'].dt.quarter
            elif col == 'week_of_year':
                df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Verify columns and row counts
        print(f"Updated shape: {df.shape}")
        df.to_csv(filepath, index=False)
        print("Dataset successfully updated in-place!")
    else:
        print("\nAll calendar features are already present. No modification needed.")

    # Re-load the updated dataset for auditing
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    print("\n--- LEAKAGE AUDIT ---")
    
    # 1. Verify Lags (Shift)
    # lag_1_sales should be equal to sales_amount shifted by 1
    lag_1_sales_ok = df['lag_1_sales'].equals(df['sales_amount'].shift(1))
    lag_7_sales_ok = df['lag_7_sales'].equals(df['sales_amount'].shift(7))
    lag_30_sales_ok = df['lag_30_sales'].equals(df['sales_amount'].shift(30))
    lag_1_qty_ok = df['lag_1_qty'].equals(df['quantity_sold'].shift(1))
    lag_7_qty_ok = df['lag_7_qty'].equals(df['quantity_sold'].shift(7))
    lag_30_qty_ok = df['lag_30_qty'].equals(df['quantity_sold'].shift(30))
    
    print(f"lag_1_sales shift check: {'PASS' if lag_1_sales_ok else 'FAIL'}")
    print(f"lag_7_sales shift check: {'PASS' if lag_7_sales_ok else 'FAIL'}")
    print(f"lag_30_sales shift check: {'PASS' if lag_30_sales_ok else 'FAIL'}")
    print(f"lag_1_qty shift check: {'PASS' if lag_1_qty_ok else 'FAIL'}")
    print(f"lag_7_qty shift check: {'PASS' if lag_7_qty_ok else 'FAIL'}")
    print(f"lag_30_qty shift check: {'PASS' if lag_30_qty_ok else 'FAIL'}")
    
    # If fails, show diff
    if not lag_1_sales_ok:
        print("Sample diff for lag_1_sales:")
        print(pd.DataFrame({'sales': df['sales_amount'], 'lag_1_sales': df['lag_1_sales'], 'shifted': df['sales_amount'].shift(1)}).head(10))

    # 2. Rolling features check
    # Check rolling_7_day_sales
    rolling_7_sales_ok = df['rolling_7_day_sales'].equals(df['sales_amount'].rolling(window=7, min_periods=1).mean())
    rolling_30_sales_ok = df['rolling_30_day_sales'].equals(df['sales_amount'].rolling(window=30, min_periods=1).mean())
    rolling_7_qty_ok = df['rolling_7_day_quantity'].equals(df['quantity_sold'].rolling(window=7, min_periods=1).mean())
    rolling_30_qty_ok = df['rolling_30_day_quantity'].equals(df['quantity_sold'].rolling(window=30, min_periods=1).mean())
    
    print(f"rolling_7_day_sales window check: {'PASS' if rolling_7_sales_ok else 'FAIL'}")
    print(f"rolling_30_day_sales window check: {'PASS' if rolling_30_sales_ok else 'FAIL'}")
    print(f"rolling_7_day_quantity window check: {'PASS' if rolling_7_qty_ok else 'FAIL'}")
    print(f"rolling_30_day_quantity window check: {'PASS' if rolling_30_qty_ok else 'FAIL'}")

    # 3. Check sales_growth_rate calculation
    # We want to see if sales_growth_rate is computed using lagged months or current month
    # Let's print sales_growth_rate for each month
    df['year_month'] = df['date'].dt.to_period('M')
    growth_by_month = df.groupby('year_month')['sales_growth_rate'].unique()
    print("\nsales_growth_rate by month:")
    for m, val in growth_by_month.items():
        print(f"  {m}: {val}")
        
    # Let's compute monthly sales totals
    monthly_sales = df.groupby('year_month')['sales_amount'].sum()
    print("\nmonthly sales totals:")
    for m, val in monthly_sales.items():
        print(f"  {m}: {val:.2f}")

    # Check if growth rate for month M matches (sales[M-1] - sales[M-2]) / sales[M-2]
    print("\nVerifying growth rate calculation logic:")
    for idx, (m, val) in enumerate(monthly_sales.items()):
        if idx >= 2:
            prev_m = monthly_sales.index[idx-1]
            prev_prev_m = monthly_sales.index[idx-2]
            computed_growth = (monthly_sales[prev_m] - monthly_sales[prev_prev_m]) / monthly_sales[prev_prev_m]
            actual_growth = df[df['year_month'] == m]['sales_growth_rate'].iloc[0]
            print(f"  Month: {m} | Actual in CSV: {actual_growth:.6f} | Computed Lagged (M-1 vs M-2): {computed_growth:.6f} | Matches: {abs(actual_growth - computed_growth) < 1e-5}")
        elif idx == 1:
            actual_growth = df[df['year_month'] == m]['sales_growth_rate'].iloc[0]
            print(f"  Month: {m} | Actual in CSV: {actual_growth:.6f} | Should be 0.0 or NaT")
            
    # 4. Check seasonality indices
    print("\n--- SEASONALITY INDEX CHECK ---")
    overall_mean = df['sales_amount'].mean()
    print(f"Overall sales amount mean: {overall_mean:.4f}")
    
    # Weekday Seasonality Index vs actual day of week mean / overall mean
    df['day_of_week_tmp'] = df['date'].dt.dayofweek
    weekday_means_computed = df.groupby('day_of_week_tmp')['sales_amount'].mean() / overall_mean
    weekday_seasonality_csv = df.groupby('day_of_week_tmp')['weekday_seasonality_index'].first()
    print("\nWeekday Seasonality Index matches computed over full dataset:")
    for d in range(7):
        print(f"  Day {d}: CSV={weekday_seasonality_csv[d]:.6f}, Computed={weekday_means_computed[d]:.6f}")
        
    # Monthly Seasonality Index vs actual month mean / overall mean
    df['month_tmp'] = df['date'].dt.month
    monthly_means_computed = df.groupby('month_tmp')['sales_amount'].mean() / overall_mean
    monthly_seasonality_csv = df.groupby('month_tmp')['monthly_seasonality_index'].first()
    print("\nMonthly Seasonality Index matches computed over full dataset:")
    for m in range(1, 13):
        if m in monthly_seasonality_csv.index:
            print(f"  Month {m}: CSV={monthly_seasonality_csv[m]:.6f}, Computed={monthly_means_computed[m]:.6f}")

if __name__ == '__main__':
    main()
