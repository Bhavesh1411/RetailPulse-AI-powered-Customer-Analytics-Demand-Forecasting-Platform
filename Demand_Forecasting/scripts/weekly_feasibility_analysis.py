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
    
    # Set date as index for resampling
    df.set_index('date', inplace=True)
    
    # Aggregate weekly. We resample by ISO week (Monday-Sunday).
    # 'W-SUN' resamples to weeks ending on Sunday. Let's check how many rows.
    weekly_df = df.resample('W-SUN').agg({
        'sales_amount': 'sum',
        'quantity_sold': 'sum'
    }).rename(columns={
        'sales_amount': 'weekly_sales_amount',
        'quantity_sold': 'weekly_quantity_sold'
    })
    
    # Date index becomes the Sunday ending date. Let's make index a column.
    weekly_df.reset_index(inplace=True)
    
    # 1. Weekly Observations & Date Range
    num_obs = len(weekly_df)
    start_date = weekly_df['date'].min()
    end_date = weekly_df['date'].max()
    
    print("--- PHASE 1: WEEKLY DATASET SUMMARY ---")
    print(f"Weekly observations: {num_obs}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Check for missing weeks: the dates should be exactly 7 days apart.
    date_diffs = weekly_df['date'].diff().dropna()
    missing_weeks = (date_diffs != pd.Timedelta(days=7)).sum()
    print(f"Missing weeks count: {missing_weeks}")
    
    # Trend and seasonality check (simple correlation with time or visual trend)
    weekly_df['time_idx'] = np.arange(len(weekly_df))
    trend_corr = weekly_df['weekly_sales_amount'].corr(weekly_df['time_idx'])
    print(f"Correlation with time index (trend): {trend_corr:.4f}")
    
    # 2. Weekly Forecasting Readiness
    print("\n--- PHASE 2: FORECASTING READINESS ---")
    print("Null values count:")
    print(weekly_df.isnull().sum())
    print("Duplicate rows:", weekly_df.duplicated(subset=['date']).sum())
    
    # Outliers: define using Z-score or IQR
    q1 = weekly_df['weekly_sales_amount'].quantile(0.25)
    q3 = weekly_df['weekly_sales_amount'].quantile(0.75)
    iqr = q3 - q1
    outliers = weekly_df[(weekly_df['weekly_sales_amount'] < (q1 - 1.5 * iqr)) | (weekly_df['weekly_sales_amount'] > (q3 + 1.5 * iqr))]
    print(f"Outliers count (IQR method): {len(outliers)}")
    if len(outliers) > 0:
        print("Outlier dates:")
        for idx, row in outliers.iterrows():
            print(f"  {row['date'].date()}: ${row['weekly_sales_amount']:,.2f}")
            
    # ADF Stationarity Test
    adf_result = adfuller(weekly_df['weekly_sales_amount'])
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"Stationary? {'YES' if adf_result[1] < 0.05 else 'NO'}")
    
    # 3. Feature Engineering
    # We will build:
    # - lag_1_weekly_sales, lag_2_weekly_sales, lag_4_weekly_sales
    # - rolling_4_week_sales_lag1
    # - weekly_sales_growth_rate
    # - weekly_sales_volatility
    # - week_of_year
    # - month
    # - quarter
    
    weekly_df['lag_1_weekly_sales'] = weekly_df['weekly_sales_amount'].shift(1)
    weekly_df['lag_2_weekly_sales'] = weekly_df['weekly_sales_amount'].shift(2)
    weekly_df['lag_4_weekly_sales'] = weekly_df['weekly_sales_amount'].shift(4)
    
    # Rolling averages over lag 1 (past weeks)
    weekly_df['rolling_4_week_sales_lag1'] = weekly_df['lag_1_weekly_sales'].rolling(window=4).mean()
    
    # Weekly sales growth rate
    epsilon = 1e-5
    weekly_df['weekly_sales_growth_rate'] = (weekly_df['lag_1_weekly_sales'] - weekly_df['lag_2_weekly_sales']) / (weekly_df['lag_2_weekly_sales'] + epsilon)
    
    # Volatility
    weekly_df['weekly_sales_volatility'] = weekly_df['lag_1_weekly_sales'].rolling(window=4).std()
    
    # Calendar features
    weekly_df['week_of_year'] = weekly_df['date'].dt.isocalendar().week.astype(int)
    # Month of the week start date (date index is Sunday ending date, so start date is index - 6 days)
    weekly_df['week_start_date'] = weekly_df['date'] - pd.Timedelta(days=6)
    weekly_df['month'] = weekly_df['week_start_date'].dt.month
    weekly_df['quarter'] = weekly_df['week_start_date'].dt.quarter
    
    # Let's drop rows with NaN values resulting from shift/rolling (first 4 rows)
    clean_weekly_df = weekly_df.dropna().copy()
    clean_weekly_df.reset_index(drop=True, inplace=True)
    
    print(f"\nClean weekly dataset rows after dropping NaNs: {len(clean_weekly_df)}")
    
    # Correlation between lag_1_weekly_sales and weekly_sales_amount
    lag1_corr = clean_weekly_df['lag_1_weekly_sales'].corr(clean_weekly_df['weekly_sales_amount'])
    print(f"Correlation between lag_1_weekly_sales and weekly_sales_amount: {lag1_corr:.4f}")
    
    # 4. Baseline Weekly Model Test
    # Features to use:
    features = [
        'lag_1_weekly_sales', 'lag_2_weekly_sales', 'lag_4_weekly_sales',
        'rolling_4_week_sales_lag1', 'weekly_sales_growth_rate', 'weekly_sales_volatility',
        'week_of_year', 'month', 'quarter'
    ]
    target = 'weekly_sales_amount'
    
    # Chronological Split
    # Since total rows = 50 (after NaNs), let's split:
    # Test = 4 weeks
    # Val = 4 weeks
    # Train = 42 weeks
    test_size = 4
    val_size = 4
    train_size = len(clean_weekly_df) - test_size - val_size
    
    train_data = clean_weekly_df.iloc[:train_size]
    val_data = clean_weekly_df.iloc[train_size:train_size+val_size]
    test_data = clean_weekly_df.iloc[-test_size:]
    
    print("\n--- PHASE 4: BASELINE WEEKLY MODEL TEST ---")
    print(f"Train period: {train_data['date'].min().date()} to {train_data['date'].max().date()} ({len(train_data)} weeks)")
    print(f"Val period:   {val_data['date'].min().date()} to {val_data['date'].max().date()} ({len(val_data)} weeks)")
    print(f"Test period:  {test_data['date'].min().date()} to {test_data['date'].max().date()} ({len(test_data)} weeks)")
    
    X_train, y_train = train_data[features], train_data[target]
    X_val, y_val = val_data[features], val_data[target]
    X_test, y_test = test_data[features], test_data[target]
    
    # XGBoost Baseline
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, early_stopping_rounds=15, random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_train_preds = xgb_model.predict(X_train)
    xgb_val_preds = xgb_model.predict(X_val)
    xgb_test_preds = xgb_model.predict(X_test)
    
    xgb_metrics_train = calculate_metrics(y_train, xgb_train_preds)
    xgb_metrics_val = calculate_metrics(y_val, xgb_val_preds)
    xgb_metrics_test = calculate_metrics(y_test, xgb_test_preds)
    
    print("\nXGBoost Weekly Baseline:")
    print(f"  Train -> MAPE: {xgb_metrics_train[0]:.2f}%, MAE: {xgb_metrics_train[1]:.2f}, RMSE: {xgb_metrics_train[2]:.2f}, R2: {xgb_metrics_train[3]:.4f}")
    print(f"  Val   -> MAPE: {xgb_metrics_val[0]:.2f}%, MAE: {xgb_metrics_val[1]:.2f}, RMSE: {xgb_metrics_val[2]:.2f}, R2: {xgb_metrics_val[3]:.4f}")
    print(f"  Test  -> MAPE: {xgb_metrics_test[0]:.2f}%, MAE: {xgb_metrics_test[1]:.2f}, RMSE: {xgb_metrics_test[2]:.2f}, R2: {xgb_metrics_test[3]:.4f}")
    
    # LightGBM Baseline
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(15, verbose=False)])
    
    lgb_train_preds = lgb_model.predict(X_train)
    lgb_val_preds = lgb_model.predict(X_val)
    lgb_test_preds = lgb_model.predict(X_test)
    
    lgb_metrics_train = calculate_metrics(y_train, lgb_train_preds)
    lgb_metrics_val = calculate_metrics(y_val, lgb_val_preds)
    lgb_metrics_test = calculate_metrics(y_test, lgb_test_preds)
    
    print("\nLightGBM Weekly Baseline:")
    print(f"  Train -> MAPE: {lgb_metrics_train[0]:.2f}%, MAE: {lgb_metrics_train[1]:.2f}, RMSE: {lgb_metrics_train[2]:.2f}, R2: {lgb_metrics_train[3]:.4f}")
    print(f"  Val   -> MAPE: {lgb_metrics_val[0]:.2f}%, MAE: {lgb_metrics_val[1]:.2f}, RMSE: {lgb_metrics_val[2]:.2f}, R2: {lgb_metrics_val[3]:.4f}")
    print(f"  Test  -> MAPE: {lgb_metrics_test[0]:.2f}%, MAE: {lgb_metrics_test[1]:.2f}, RMSE: {lgb_metrics_test[2]:.2f}, R2: {lgb_metrics_test[3]:.4f}")

if __name__ == '__main__':
    main()
