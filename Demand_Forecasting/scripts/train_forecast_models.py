import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    # Avoid divide by zero in MAPE by masking out zero values
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
    print("Loading dataset...")
    filepath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/daily_sales_forecast_features.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort chronologically just in case
    df = df.sort_values('date').reset_index(drop=True)
    
    features = [
        'lag_1_sales', 'lag_1_qty', 'lag_7_sales', 'lag_7_qty', 'lag_30_sales', 'lag_30_qty',
        'weekday_seasonality_index', 'sales_growth_rate', 'day_of_week', 'month', 'quarter', 'week_of_year',
        'rolling_7_day_sales_lag1', 'rolling_30_day_sales_lag1', 'rolling_7_day_quantity_lag1',
        'rolling_30_day_quantity_lag1', 'sales_volatility_30_day_lag1', 'demand_momentum_lag1',
        'quantity_momentum_lag1'
    ]
    target_sales = 'sales_amount'
    target_qty = 'quantity_sold'
    
    # Drop rows with NaN in features (due to lag_30)
    valid_df = df.dropna(subset=features).copy()
    valid_df.reset_index(drop=True, inplace=True)
    
    total_rows = len(valid_df)
    test_size = 30
    val_size = 30
    train_size = total_rows - test_size - val_size
    
    train_df = valid_df.iloc[:train_size]
    val_df = valid_df.iloc[train_size:train_size+val_size]
    test_df = valid_df.iloc[-test_size:]
    
    print("\n==================================================")
    print("PHASE 2 – TIME SERIES SPLIT")
    print("==================================================")
    print(f"Training period:   {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} days)")
    print(f"Validation period: {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(val_df)} days)")
    print(f"Testing period:    {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} days)")
    
    X_train, y_train_sales, y_train_qty = train_df[features], train_df[target_sales], train_df[target_qty]
    X_val, y_val_sales, y_val_qty = val_df[features], val_df[target_sales], val_df[target_qty]
    X_test, y_test_sales, y_test_qty = test_df[features], test_df[target_sales], test_df[target_qty]
    
    # --------------------------------------------------
    # PHASE 3 & 4 – MODEL TRAINING & EVALUATION
    # --------------------------------------------------
    print("\n==================================================")
    print("PHASE 3 & 4 – MODEL TRAINING & EVALUATION")
    print("==================================================")
    
    # XGBoost
    print("Training XGBoost Regressor...")
    xgb_sales = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, early_stopping_rounds=20, random_state=42)
    xgb_sales.fit(X_train, y_train_sales, eval_set=[(X_val, y_val_sales)], verbose=False)
    
    xgb_qty = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, early_stopping_rounds=20, random_state=42)
    xgb_qty.fit(X_train, y_train_qty, eval_set=[(X_val, y_val_qty)], verbose=False)
    
    xgb_train_preds = xgb_sales.predict(X_train)
    xgb_val_preds = xgb_sales.predict(X_val)
    xgb_test_preds = xgb_sales.predict(X_test)
    
    xgb_mape_train, xgb_mae_train, xgb_rmse_train, xgb_r2_train = calculate_metrics(y_train_sales, xgb_train_preds)
    xgb_mape_val, xgb_mae_val, xgb_rmse_val, xgb_r2_val = calculate_metrics(y_val_sales, xgb_val_preds)
    xgb_mape_test, xgb_mae_test, xgb_rmse_test, xgb_r2_test = calculate_metrics(y_test_sales, xgb_test_preds)
    
    print("\nXGBoost Results (Sales):")
    print(f"  Train -> MAPE: {xgb_mape_train:.2f}%, MAE: {xgb_mae_train:.2f}, RMSE: {xgb_rmse_train:.2f}, R2: {xgb_r2_train:.4f}")
    print(f"  Val   -> MAPE: {xgb_mape_val:.2f}%, MAE: {xgb_mae_val:.2f}, RMSE: {xgb_rmse_val:.2f}, R2: {xgb_r2_val:.4f}")
    print(f"  Test  -> MAPE: {xgb_mape_test:.2f}%, MAE: {xgb_mae_test:.2f}, RMSE: {xgb_rmse_test:.2f}, R2: {xgb_r2_test:.4f}")
    
    # LightGBM
    print("\nTraining LightGBM Regressor...")
    lgb_sales = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    lgb_sales.fit(X_train, y_train_sales, eval_set=[(X_val, y_val_sales)], eval_metric='mape', callbacks=[lgb.early_stopping(20, verbose=False)])
    
    lgb_qty = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    lgb_qty.fit(X_train, y_train_qty, eval_set=[(X_val, y_val_qty)], eval_metric='mape', callbacks=[lgb.early_stopping(20, verbose=False)])
    
    lgb_train_preds = lgb_sales.predict(X_train)
    lgb_val_preds = lgb_sales.predict(X_val)
    lgb_test_preds = lgb_sales.predict(X_test)
    
    lgb_mape_train, lgb_mae_train, lgb_rmse_train, lgb_r2_train = calculate_metrics(y_train_sales, lgb_train_preds)
    lgb_mape_val, lgb_mae_val, lgb_rmse_val, lgb_r2_val = calculate_metrics(y_val_sales, lgb_val_preds)
    lgb_mape_test, lgb_mae_test, lgb_rmse_test, lgb_r2_test = calculate_metrics(y_test_sales, lgb_test_preds)
    
    print("\nLightGBM Results (Sales):")
    print(f"  Train -> MAPE: {lgb_mape_train:.2f}%, MAE: {lgb_mae_train:.2f}, RMSE: {lgb_rmse_train:.2f}, R2: {lgb_r2_train:.4f}")
    print(f"  Val   -> MAPE: {lgb_mape_val:.2f}%, MAE: {lgb_mae_val:.2f}, RMSE: {lgb_rmse_val:.2f}, R2: {lgb_r2_val:.4f}")
    print(f"  Test  -> MAPE: {lgb_mape_test:.2f}%, MAE: {lgb_mae_test:.2f}, RMSE: {lgb_rmse_test:.2f}, R2: {lgb_r2_test:.4f}")
    
    # --------------------------------------------------
    # PHASE 5 – MODEL COMPARISON
    # --------------------------------------------------
    print("\n==================================================")
    print("PHASE 5 – MODEL COMPARISON")
    print("==================================================")
    if xgb_mape_test <= lgb_mape_test:
        best_model_name = "XGBoost"
        best_model_sales = xgb_sales
        best_model_qty = xgb_qty
        best_mape = xgb_mape_test
    else:
        best_model_name = "LightGBM"
        best_model_sales = lgb_sales
        best_model_qty = lgb_qty
        best_mape = lgb_mape_test
        
    print(f"Winner: {best_model_name} (Test MAPE: {best_mape:.2f}%)")
    
    # --------------------------------------------------
    # PHASE 6 – 30 DAY FORECAST
    # --------------------------------------------------
    print("\n==================================================")
    print("PHASE 6 – 30 DAY FORECAST")
    print("==================================================")
    
    # To forecast iteratively, we need the last known 30 days of data.
    # We will append predictions day by day.
    history_df = df.copy()
    
    last_date = history_df['date'].max()
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
    
    forecasts = []
    
    for f_date in forecast_dates:
        # Create a new row
        new_row = pd.DataFrame({'date': [f_date]})
        new_row['day_of_week'] = f_date.dayofweek
        new_row['month'] = f_date.month
        new_row['quarter'] = f_date.quarter
        new_row['week_of_year'] = f_date.isocalendar().week
        
        # Calculate Lags from history
        new_row['lag_1_sales'] = history_df['sales_amount'].iloc[-1]
        new_row['lag_7_sales'] = history_df['sales_amount'].iloc[-7]
        new_row['lag_30_sales'] = history_df['sales_amount'].iloc[-30]
        
        new_row['lag_1_qty'] = history_df['quantity_sold'].iloc[-1]
        new_row['lag_7_qty'] = history_df['quantity_sold'].iloc[-7]
        new_row['lag_30_qty'] = history_df['quantity_sold'].iloc[-30]
        
        # Rolling Averages (up to yesterday, which corresponds to _lag1)
        new_row['rolling_7_day_sales_lag1'] = history_df['sales_amount'].iloc[-7:].mean()
        new_row['rolling_30_day_sales_lag1'] = history_df['sales_amount'].iloc[-30:].mean()
        new_row['rolling_7_day_quantity_lag1'] = history_df['quantity_sold'].iloc[-7:].mean()
        new_row['rolling_30_day_quantity_lag1'] = history_df['quantity_sold'].iloc[-30:].mean()
        
        # Volatility
        new_row['sales_volatility_30_day_lag1'] = history_df['sales_amount'].iloc[-30:].std()
        
        # Momentum
        if new_row['rolling_30_day_sales_lag1'].iloc[0] > 0:
            new_row['demand_momentum_lag1'] = new_row['rolling_7_day_sales_lag1'] / new_row['rolling_30_day_sales_lag1']
        else:
            new_row['demand_momentum_lag1'] = 1.0
            
        if new_row['rolling_30_day_quantity_lag1'].iloc[0] > 0:
            new_row['quantity_momentum_lag1'] = new_row['rolling_7_day_quantity_lag1'] / new_row['rolling_30_day_quantity_lag1']
        else:
            new_row['quantity_momentum_lag1'] = 1.0
            
        # Growth Rate (M-1 vs M-2 logic)
        curr_period = pd.Period(f_date, freq='M')
        # Filter history up to M-1
        hist_M1 = history_df[history_df['date'].dt.to_period('M') == curr_period - 1]
        hist_M2 = history_df[history_df['date'].dt.to_period('M') == curr_period - 2]
        sum_M1 = hist_M1['sales_amount'].sum() if len(hist_M1) > 0 else 0
        sum_M2 = hist_M2['sales_amount'].sum() if len(hist_M2) > 0 else 0
        
        if sum_M2 > 0:
            new_row['sales_growth_rate'] = (sum_M1 - sum_M2) / sum_M2
        else:
            new_row['sales_growth_rate'] = 0.0
            
        # Weekday Seasonality Index (can use the last known for that day_of_week)
        # Using the mapping from the training data
        wd = int(new_row['day_of_week'].iloc[0])
        wd_idx = history_df[history_df['day_of_week'] == wd]['weekday_seasonality_index'].iloc[-1]
        new_row['weekday_seasonality_index'] = wd_idx
        
        # Predict
        X_pred = new_row[features]
        pred_sales = best_model_sales.predict(X_pred)[0]
        pred_qty = best_model_qty.predict(X_pred)[0]
        
        # Prevent negative forecasts
        pred_sales = max(0, pred_sales)
        pred_qty = max(0, pred_qty)
        
        # Append to history
        new_row['sales_amount'] = pred_sales
        new_row['quantity_sold'] = pred_qty
        
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        forecasts.append({'date': f_date.date(), 'sales_forecast': pred_sales})
        
    print("\nForecast Sample (First 5 days):")
    for f in forecasts[:5]:
        print(f"  {f['date']}: ${f['sales_forecast']:,.2f}")
    
    total_forecast = sum(f['sales_forecast'] for f in forecasts)
    print(f"\nTotal 30-Day Demand Forecast: ${total_forecast:,.2f}")
    
    # --------------------------------------------------
    # PHASE 7 – REQUIREMENT CHECK
    # --------------------------------------------------
    print("\n==================================================")
    print("PHASE 7 – REQUIREMENT CHECK")
    print("==================================================")
    print(f"1. Best Model: {best_model_name}")
    print(f"2. Final Test MAPE: {best_mape:.2f}%")
    print(f"3. Requirement Achieved (MAPE <= 12%): {'YES' if best_mape <= 12.0 else 'NO'}")
    print(f"4. Demand Forecasting Approved: {'YES' if best_mape <= 12.0 else 'NO'}")

if __name__ == '__main__':
    main()
