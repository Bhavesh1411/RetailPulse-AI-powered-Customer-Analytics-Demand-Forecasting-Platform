import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        mape = 0
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mape, mae, rmse, r2

def main():
    filepath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/daily_sales_forecast_features.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Let's count days per week when resampling
    df['dummy_count'] = 1
    weekly_raw = df.resample('W-SUN', on='date').agg({
        'sales_amount': 'sum',
        'quantity_sold': 'sum',
        'dummy_count': 'sum'
    })
    
    print("Weekly Raw Aggregation (First 3 and Last 3 rows):")
    print(weekly_raw.head(3))
    print(weekly_raw.tail(3))
    
    # Save the weekly aggregated dataset
    weekly_dest = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/weekly_sales_forecast_features.csv')
    
    # We will resample and create the features
    # Let's use W-SUN and clean up partial weeks if they distort results.
    # The first week (2009-12-06) has 6 days, which is almost a full week.
    # The last week (2010-12-12) has only 4 days, which is a partial week.
    # Let's keep them but note the count.
    
    weekly_df = weekly_raw.rename(columns={
        'sales_amount': 'weekly_sales_amount',
        'quantity_sold': 'weekly_quantity_sold'
    })
    weekly_df.reset_index(inplace=True)
    
    # Save to CSV
    weekly_df[['date', 'weekly_sales_amount', 'weekly_quantity_sold']].to_csv(weekly_dest, index=False)
    print(f"\nSaved weekly aggregated dataset to {weekly_dest}")

if __name__ == '__main__':
    main()
