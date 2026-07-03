import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def main():
    filepath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/daily_sales_forecast_features.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Resample weekly
    weekly_df = df.resample('W-SUN', on='date').agg({
        'sales_amount': 'sum',
        'quantity_sold': 'sum'
    }).rename(columns={
        'sales_amount': 'weekly_sales_amount',
        'quantity_sold': 'weekly_quantity_sold'
    }).reset_index()
    
    # Build features
    weekly_df['lag_1_weekly_sales'] = weekly_df['weekly_sales_amount'].shift(1)
    weekly_df['lag_2_weekly_sales'] = weekly_df['weekly_sales_amount'].shift(2)
    weekly_df['lag_4_weekly_sales'] = weekly_df['weekly_sales_amount'].shift(4)
    weekly_df['rolling_4_week_sales_lag1'] = weekly_df['lag_1_weekly_sales'].rolling(window=4).mean()
    weekly_df['weekly_sales_growth_rate'] = (weekly_df['lag_1_weekly_sales'] - weekly_df['lag_2_weekly_sales']) / (weekly_df['lag_2_weekly_sales'] + 1e-5)
    weekly_df['weekly_sales_volatility'] = weekly_df['lag_1_weekly_sales'].rolling(window=4).std()
    weekly_df['week_of_year'] = weekly_df['date'].dt.isocalendar().week.astype(int)
    weekly_df['week_start_date'] = weekly_df['date'] - pd.Timedelta(days=6)
    weekly_df['month'] = weekly_df['week_start_date'].dt.month
    weekly_df['quarter'] = weekly_df['week_start_date'].dt.quarter
    
    clean_weekly_df = weekly_df.dropna().copy()
    clean_weekly_df.reset_index(drop=True, inplace=True)
    
    features = [
        'lag_1_weekly_sales', 'lag_2_weekly_sales', 'lag_4_weekly_sales',
        'rolling_4_week_sales_lag1', 'weekly_sales_growth_rate', 'weekly_sales_volatility',
        'week_of_year', 'month', 'quarter'
    ]
    target = 'weekly_sales_amount'
    
    test_size = 4
    val_size = 4
    train_size = len(clean_weekly_df) - test_size - val_size
    
    train_data = clean_weekly_df.iloc[:train_size]
    val_data = clean_weekly_df.iloc[train_size:train_size+val_size]
    test_data = clean_weekly_df.iloc[-test_size:]
    
    X_train, y_train = train_data[features], train_data[target]
    X_val, y_val = val_data[features], val_data[target]
    X_test, y_test = test_data[features], test_data[target]
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, early_stopping_rounds=15, random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_test_preds = xgb_model.predict(X_test)
    
    print("XGBoost Test Predictions vs Actuals:")
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        pred = xgb_test_preds[i]
        actual = y_test.iloc[i]
        error = actual - pred
        pct_err = abs(error) / actual * 100
        print(f"  Date: {row['date'].date()} | Actual: ${actual:,.2f} | Predicted: ${pred:,.2f} | Error: ${error:,.2f} | MAPE: {pct_err:.2f}%")

if __name__ == '__main__':
    main()
