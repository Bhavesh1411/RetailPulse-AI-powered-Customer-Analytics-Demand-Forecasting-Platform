import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to Agg to avoid UI issues
import matplotlib
matplotlib.use('Agg')

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
    print("Loading dataset...")
    filepath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/daily_sales_forecast_features.csv')
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
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
    
    # Drop rows with NaN in features
    valid_df = df.dropna(subset=features).copy()
    valid_df.reset_index(drop=True, inplace=True)
    
    total_rows = len(valid_df)
    test_size = 30
    val_size = 30
    train_size = total_rows - test_size - val_size
    
    train_df = valid_df.iloc[:train_size]
    val_df = valid_df.iloc[train_size:train_size+val_size]
    test_df = valid_df.iloc[-test_size:]
    
    X_train, y_train_sales = train_df[features], train_df[target_sales]
    X_val, y_val_sales = val_df[features], val_df[target_sales]
    X_test, y_test_sales = test_df[features], test_df[target_sales]
    
    # Train LightGBM model
    lgb_sales = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    lgb_sales.fit(X_train, y_train_sales, eval_set=[(X_val, y_val_sales)], eval_metric='mape', callbacks=[lgb.early_stopping(20, verbose=False)])
    
    # Predict
    train_preds = lgb_sales.predict(X_train)
    val_preds = lgb_sales.predict(X_val)
    test_preds = lgb_sales.predict(X_test)
    
    # Phase 1: Model Diagnostic Report
    mape_train, mae_train, rmse_train, r2_train = calculate_metrics(y_train_sales, train_preds)
    mape_val, mae_val, rmse_val, r2_val = calculate_metrics(y_val_sales, val_preds)
    mape_test, mae_test, rmse_test, r2_test = calculate_metrics(y_test_sales, test_preds)
    
    print("\n--- PHASE 1: DIAGNOSTIC METRICS ---")
    print(f"Train MAPE: {mape_train:.2f}%, MAE: {mae_train:.2f}, RMSE: {rmse_train:.2f}, R2: {r2_train:.4f}")
    print(f"Val MAPE: {mape_val:.2f}%, MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}, R2: {r2_val:.4f}")
    print(f"Test MAPE: {mape_test:.2f}%, MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, R2: {r2_test:.4f}")
    
    # Phase 2: Feature Importance
    importances = lgb_sales.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n--- PHASE 2: FEATURE IMPORTANCES (GAIN/SPLIT) ---")
    print("Top Features:")
    for idx in indices[:10]:
        print(f"  {features[idx]}: {importances[idx]}")
    print("Bottom Features:")
    for idx in indices[-10:]:
        print(f"  {features[idx]}: {importances[idx]}")
        
    # Phase 3: SHAP Analysis
    print("\n--- PHASE 3: SHAP ANALYSIS ---")
    explainer = shap.TreeExplainer(lgb_sales)
    shap_values = explainer.shap_values(X_test)
    
    # Mean absolute SHAP values for each feature
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    shap_indices = np.argsort(mean_shap)[::-1]
    
    print("SHAP Global Feature Importance (on Test set):")
    for idx in shap_indices:
        print(f"  {features[idx]}: {mean_shap[idx]:.4f}")
        
    # Directional impact (correlation of SHAP value with feature value)
    print("\nDirectional Impact (SHAP value vs Feature value correlation):")
    for idx in shap_indices[:10]:
        feat_vals = X_test[features[idx]].values
        s_vals = shap_values[:, idx]
        # Avoid correlation calculation if variance is zero
        if np.std(feat_vals) > 0 and np.std(s_vals) > 0:
            corr = np.corrcoef(feat_vals, s_vals)[0, 1]
            direction = "Positive" if corr > 0 else "Negative"
            print(f"  {features[idx]}: Corr={corr:.4f} ({direction} Impact)")
        else:
            print(f"  {features[idx]}: N/A (zero variance)")
            
    # Phase 4: Residual Analysis
    print("\n--- PHASE 4: RESIDUAL ANALYSIS ---")
    residuals = y_test_sales - test_preds
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    
    print(f"Residual Mean: {res_mean:.2f}")
    print(f"Residual Std Dev: {res_std:.2f}")
    
    # Check for bias
    over_count = np.sum(residuals < 0)  # pred > actual -> overprediction
    under_count = np.sum(residuals > 0) # pred < actual -> underprediction
    print(f"Overpredictions (Pred > Actual): {over_count} days")
    print(f"Underpredictions (Pred < Actual): {under_count} days")
    
    # Largest misses
    test_dates = test_df['date'].values
    miss_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test_sales,
        'Predicted': test_preds,
        'Residual': residuals,
        'AbsResidual': np.abs(residuals)
    })
    largest_misses = miss_df.sort_values(by='AbsResidual', ascending=False).head(5)
    print("\nLargest Forecasting Misses:")
    for i, row in largest_misses.iterrows():
        print(f"  Date: {pd.to_datetime(row['Date']).date()} | Actual: ${row['Actual']:,.2f} | Predicted: ${row['Predicted']:,.2f} | Error: ${row['Residual']:,.2f}")
        
    # Phase 5: Actual vs Predicted Plots
    print("\n--- PHASE 5: GENERATING PLOTS ---")
    artifact_dir = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\ef499792-a9c3-4bd3-a984-6da54fe0fe7d"
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['date'], y_test_sales, label='Actual Sales', marker='o', color='blue')
    plt.plot(test_df['date'], test_preds, label='Predicted Sales', marker='x', linestyle='--', color='red')
    plt.title('Actual vs Predicted Sales (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Sales Amount ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path1 = os.path.join(artifact_dir, 'actual_vs_predicted.png')
    plt.savefig(plot_path1)
    plt.close()
    print(f"Saved plot: {plot_path1}")
    
    # Residuals plot over time
    plt.figure(figsize=(12, 6))
    plt.bar(test_df['date'], residuals, color='purple', alpha=0.6)
    plt.axhline(0, color='black', linestyle='-')
    plt.title('Residuals Over Time (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Residual ($)')
    plt.grid(True)
    plt.tight_layout()
    plot_path2 = os.path.join(artifact_dir, 'residuals_over_time.png')
    plt.savefig(plot_path2)
    plt.close()
    print(f"Saved plot: {plot_path2}")

if __name__ == '__main__':
    main()
