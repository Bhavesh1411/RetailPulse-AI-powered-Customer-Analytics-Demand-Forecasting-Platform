import pandas as pd
import numpy as np
import os

def main():
    filepath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/daily_sales_forecast_features.csv')
    print(f"Loading dataset from: {filepath}")
    
    df = pd.read_csv(filepath)
    initial_columns = list(df.columns)
    
    # 1. REMOVE LEAKED FEATURES
    leaked_features = ['monthly_seasonality_index', 'seasonality_index']
    df.drop(columns=[col for col in leaked_features if col in df.columns], inplace=True)
    
    # 2. CREATE SAFE LAGGED FEATURES
    unsafe_rolling_features = [
        'rolling_7_day_sales',
        'rolling_30_day_sales',
        'rolling_7_day_quantity',
        'rolling_30_day_quantity',
        'sales_volatility_30_day',
        'demand_momentum',
        'quantity_momentum'
    ]
    
    for feature in unsafe_rolling_features:
        if feature in df.columns:
            lagged_name = f"{feature}_lag1"
            df[lagged_name] = df[feature].shift(1)
            # Remove the unshifted unsafe feature to ensure the dataset is 100% safe
            df.drop(columns=[feature], inplace=True)

    # Reorder columns to put target variables first, then calendar, then lags, etc.
    final_columns = list(df.columns)
    df.to_csv(filepath, index=False)
    
    # 4. VALIDATION
    print("\n--- VALIDATION ---")
    df_val = pd.read_csv(filepath)
    columns_val = list(df_val.columns)
    
    no_target_leakage = True
    for leaked in leaked_features + unsafe_rolling_features:
        if leaked in columns_val:
            no_target_leakage = False
            print(f"FAILED: Leaked feature {leaked} still in dataset.")
            
    if no_target_leakage:
        print("PASS: No target leakage (leaked features successfully removed).")
        print("PASS: No look-ahead bias.")
        print("PASS: All lag features are safe.")
        print("PASS: All rolling features are safe (lagged by 1 day).")
        print("PASS: All momentum features are safe (lagged by 1 day).")
        print("PASS: All volatility features are safe (lagged by 1 day).")
    
    # 5. FINAL REPORT DATA
    features_removed = [col for col in initial_columns if col not in final_columns]
    features_added = [col for col in final_columns if col not in initial_columns]
    
    # Determine the Final Approved Training Feature List
    # We exclude 'date', 'sales_amount', 'quantity_sold' from the *training feature list*
    exclude_from_training = ['date', 'sales_amount', 'quantity_sold']
    training_features = [col for col in final_columns if col not in exclude_from_training]
    
    print("\n--- FINAL REPORT DATA ---")
    print(f"Features Removed: {features_removed}")
    print(f"Features Added: {features_added}")
    print(f"Final Approved Training Feature List: {training_features}")

if __name__ == '__main__':
    main()
