import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import datetime
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
    print("==================================================")
    print("PHASE 1 & 2 - LOAD AND ADD HOLIDAY FEATURES")
    print("==================================================")
    filepath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/weekly_sales_forecast_features.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Existing features
    df['lag_1_weekly_sales'] = df['weekly_sales_amount'].shift(1)
    df['lag_2_weekly_sales'] = df['weekly_sales_amount'].shift(2)
    df['lag_4_weekly_sales'] = df['weekly_sales_amount'].shift(4)
    df['rolling_4_week_sales_lag1'] = df['lag_1_weekly_sales'].rolling(window=4).mean()
    df['weekly_sales_growth_rate'] = (df['lag_1_weekly_sales'] - df['lag_2_weekly_sales']) / (df['lag_2_weekly_sales'] + 1e-5)
    df['weekly_sales_volatility'] = df['lag_1_weekly_sales'].rolling(window=4).std()
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    df['week_start_date'] = df['date'] - pd.Timedelta(days=6)
    df['month'] = df['week_start_date'].dt.month
    df['quarter'] = df['week_start_date'].dt.quarter
    
    # New Holiday Features
    def get_weeks_until_xmas(d):
        xmas = datetime(d.year, 12, 25)
        diff = (xmas.date() - d.date()).days
        if diff < 0:
            return 0
        return diff // 7

    def get_weeks_since_xmas(d):
        prev_xmas = datetime(d.year - 1, 12, 25)
        diff = (d.date() - prev_xmas.date()).days
        return diff // 7
        
    df['weeks_until_christmas'] = df['date'].apply(get_weeks_until_xmas)
    df['weeks_since_christmas'] = df['date'].apply(get_weeks_since_xmas)
    df['q4_flag'] = (df['quarter'] == 4).astype(int)
    df['year_end_flag'] = (df['month'] == 12).astype(int)
    df['holiday_proximity_score'] = 1.0 / (df['weeks_until_christmas'] + 1)
    
    # Save the updated dataset back (in place)
    df.to_csv(filepath, index=False)
    print(f"Added holiday features and saved directly to {filepath}")
    
    # Drop NAs
    clean_df = df.dropna().copy()
    clean_df.reset_index(drop=True, inplace=True)
    
    features = [
        'lag_1_weekly_sales', 'lag_2_weekly_sales', 'lag_4_weekly_sales',
        'rolling_4_week_sales_lag1', 'weekly_sales_growth_rate', 'weekly_sales_volatility',
        'week_of_year', 'month', 'quarter',
        'weeks_until_christmas', 'weeks_since_christmas', 'q4_flag', 
        'holiday_proximity_score', 'year_end_flag'
    ]
    target = 'weekly_sales_amount'
    
    test_size = 4
    val_size = 4
    train_size = len(clean_df) - test_size - val_size
    
    train_data = clean_df.iloc[:train_size]
    val_data = clean_df.iloc[train_size:train_size+val_size]
    test_data = clean_df.iloc[-test_size:]
    
    X_train, y_train = train_data[features], train_data[target]
    X_val, y_val = val_data[features], val_data[target]
    X_test, y_test = test_data[features], test_data[target]
    
    # Partial Week Audit (Baseline)
    print("\n==================================================")
    print("PHASE 1 - PARTIAL WEEK CORRECTION AUDIT")
    print("==================================================")
    base_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    # Train on existing features only to match previous baseline
    base_features = [f for f in features if f not in ['weeks_until_christmas', 'weeks_since_christmas', 'q4_flag', 'holiday_proximity_score', 'year_end_flag']]
    base_xgb.fit(X_train[base_features], y_train, eval_set=[(X_val[base_features], y_val)], verbose=False)
    
    base_test_preds = base_xgb.predict(X_test[base_features])
    mape_incl, _, _, _ = calculate_metrics(y_test, base_test_preds)
    mape_excl, _, _, _ = calculate_metrics(y_test.iloc[:-1], base_test_preds[:-1])
    
    print(f"Final test week date: {test_data.iloc[-1]['date'].date()} | Actual Sales: ${y_test.iloc[-1]:,.2f}")
    print(f"Is partial week? YES (Only 4 days: Dec 6 - Dec 9)")
    print(f"Baseline Test MAPE (Including partial week): {mape_incl:.2f}%")
    print(f"Baseline Test MAPE (Excluding partial week): {mape_excl:.2f}%")
    
    print("\n==================================================")
    print("PHASE 3 - FEATURE VALIDATION")
    print("==================================================")
    print("Correlation with weekly_sales_amount:")
    for f in ['weeks_until_christmas', 'weeks_since_christmas', 'q4_flag', 'holiday_proximity_score', 'year_end_flag']:
        corr = clean_df[f].corr(clean_df['weekly_sales_amount'])
        print(f"  {f}: {corr:.4f}")
        
    print("\n==================================================")
    print("PHASE 4 - OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("==================================================")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        mape, _, _, _ = calculate_metrics(y_val, preds)
        return mape
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    print(f"Best Optuna Trial: {study.best_trial.number}")
    print(f"Best Validation MAPE: {study.best_value:.2f}%")
    best_params = study.best_params
    print("Best Params:", best_params)
    
    print("\n==================================================")
    print("PHASE 5 - RETRAIN & EVALUATION")
    print("==================================================")
    best_params['random_state'] = 42
    opt_model = xgb.XGBRegressor(**best_params)
    opt_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    train_preds = opt_model.predict(X_train)
    val_preds = opt_model.predict(X_val)
    test_preds = opt_model.predict(X_test)
    
    opt_train = calculate_metrics(y_train, train_preds)
    opt_val = calculate_metrics(y_val, val_preds)
    opt_test_incl = calculate_metrics(y_test, test_preds)
    opt_test_excl = calculate_metrics(y_test.iloc[:-1], test_preds[:-1])
    
    print(f"Optimized Train -> MAPE: {opt_train[0]:.2f}%, MAE: {opt_train[1]:.2f}, RMSE: {opt_train[2]:.2f}, R2: {opt_train[3]:.4f}")
    print(f"Optimized Val   -> MAPE: {opt_val[0]:.2f}%, MAE: {opt_val[1]:.2f}, RMSE: {opt_val[2]:.2f}, R2: {opt_val[3]:.4f}")
    print(f"Optimized Test (Incl Partial Week) -> MAPE: {opt_test_incl[0]:.2f}%, MAE: {opt_test_incl[1]:.2f}, RMSE: {opt_test_incl[2]:.2f}, R2: {opt_test_incl[3]:.4f}")
    print(f"Optimized Test (Excl Partial Week) -> MAPE: {opt_test_excl[0]:.2f}%, MAE: {opt_test_excl[1]:.2f}, RMSE: {opt_test_excl[2]:.2f}, R2: {opt_test_excl[3]:.4f}")
    
    print("\nFeature Importances (Optimized Model):")
    importances = opt_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(5):
        print(f"  {features[indices[i]]}: {importances[indices[i]]:.4f}")

if __name__ == '__main__':
    main()
