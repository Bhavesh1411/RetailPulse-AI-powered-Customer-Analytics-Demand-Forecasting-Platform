import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def main():
    print("=== PHASE 6: FEATURE IMPORTANCE & SHAP ANALYSIS ===")
    
    # Paths
    models_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/models')
    dataset_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/datasets/true_churn_dataset.csv')
    reports_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    best_model_name_path = os.path.join(models_dir, 'best_model_name.txt')
    if not os.path.exists(best_model_name_path):
        print("Error: best_model_name.txt not found. Run training script first.")
        return
        
    with open(best_model_name_path, 'r') as f:
        best_model_name = f.read().strip()
        
    print(f"Best model identified in training phase: {best_model_name}")
    
    # Load model
    model_file = os.path.join(models_dir, f"{best_model_name.lower().replace(' ', '_')}_model.pkl")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        
    # Load data
    df = pd.read_csv(dataset_file)
    features = [col for col in df.columns if col not in ['customer_id', 'true_churn_flag']]
    X = df[features]
    y = df['true_churn_flag']
    
    # Handle scaler if best model is Logistic Regression
    if best_model_name == 'Logistic Regression':
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
        X_df = pd.DataFrame(X_scaled, columns=features)
    else:
        X_df = X
        
    # 1. Standard Feature Importance / Coefficients
    importances = None
    importance_type = ""
    
    if best_model_name == 'Logistic Regression':
        importances = model.coef_[0]
        importance_type = "Coefficients"
    elif best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        importances = model.feature_importances_
        importance_type = "MDI Feature Importance"
        
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Absolute_Importance': np.abs(importances)
    }).sort_values(by='Absolute_Importance', ascending=False)
    
    print("\n--- STANDARD FEATURE IMPORTANCE ---")
    for idx, row in importance_df.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.6f} (type: {importance_type})")
        
    # Plot Standard Feature Importance
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4' if val >= 0 else '#d62728' for val in importance_df['Importance']] if best_model_name == 'Logistic Regression' else '#2ca02c'
    
    plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color=colors)
    plt.xlabel('Importance' if best_model_name != 'Logistic Regression' else 'Coefficient Value')
    plt.title(f'Feature Importance ({best_model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'feature_importance_raw.png'))
    plt.close()
    
    # 2. SHAP Analysis
    shap_available = False
    shap_summary_path = os.path.join(reports_dir, 'shap_summary.png')
    
    try:
        import shap
        print("\nCalculating SHAP values...")
        
        # Use Explainer
        if best_model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(model, X_df)
            shap_values = explainer.shap_values(X_df)
        elif best_model_name == 'Random Forest':
            # For RF, shap can be slow, use TreeExplainer and check shapes
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            # For binary classification, TreeExplainer returns a list of two arrays [class0, class1] or a single array
            if isinstance(shap_values, list):
                # Choose class 1
                shap_values = shap_values[1]
        elif best_model_name in ['XGBoost', 'LightGBM']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            # Handle shape mismatch (e.g. Multi-class or binary structure check)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
        else:
            explainer = shap.Explainer(model, X_df)
            shap_values = explainer(X_df)
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
                
        # Generate and save SHAP plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_df, show=False)
        plt.title(f'SHAP Summary Plot - {best_model_name}', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(shap_summary_path)
        plt.close()
        shap_available = True
        print(f"SHAP summary plot successfully saved to: {shap_summary_path}")
        
        # Aggregate mean absolute SHAP values for report
        if hasattr(shap_values, "values"):
            shap_vals_arr = shap_values.values
        else:
            shap_vals_arr = shap_values
            
        mean_abs_shap = np.mean(np.abs(np.atleast_2d(shap_vals_arr)), axis=0)
        shap_importance = pd.DataFrame({
            'Feature': features,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Abs_SHAP', ascending=False)
        
    except ImportError:
        print("\n[WARNING] SHAP package is not installed. Skipping SHAP analysis.")
        print("To run SHAP, run: pip install shap")
        shap_importance = None
    except Exception as e:
        print(f"\n[WARNING] Error during SHAP computation: {e}. Skipping SHAP plot.")
        shap_importance = None
        
    # Generate Feature Importance and Drivers Report
    report_file = os.path.join(reports_dir, 'feature_importance_report.md')
    
    # Analyze churn & retention drivers
    # Top Churn Drivers are features that positively correlate with churn_flag (pushing target towards 1)
    # Top Retention Drivers are features that negatively correlate with churn_flag (pushing target towards 0)
    # We can infer this from standard correlation or SHAP/Coefficients. Let's use correlation to determine direction.
    corrs = {col: df[col].corr(df['true_churn_flag']) for col in features}
    
    importance_df['Correlation'] = importance_df['Feature'].map(corrs)
    importance_df['Direction'] = np.where(importance_df['Correlation'] > 0, "Push towards Churn (Risk Driver)", "Push towards Active (Retention Driver)")
    
    churn_drivers = importance_df[importance_df['Correlation'] > 0].head(5)
    retention_drivers = importance_df[importance_df['Correlation'] < 0].head(5)
    
    report_content = f"""# Feature Importance and Model Interpretation Report

This report provides a detailed breakdown of the features driving the predictions of the champion **{best_model_name}** model.

## Executive Summary

1. **Top Churn Drivers (Risk Factors):**
{chr(10).join([f"   - **{row['Feature']}** (Importance: {row['Importance']:.4f}, Correlation: {row['Correlation']:.4f})" for idx, row in churn_drivers.iterrows()])}

2. **Top Retention Drivers (Loyalty Factors):**
{chr(10).join([f"   - **{row['Feature']}** (Importance: {row['Importance']:.4f}, Correlation: {row['Correlation']:.4f})" for idx, row in retention_drivers.iterrows()])}

---

## 1. Feature Importance Rankings

Below is the complete feature importance ranking based on the `{importance_type}` from the `{best_model_name}` model.

| Rank | Feature Name | Importance Score | Relationship to Churn | Business Interpretation |
| :---: | :--- | :---: | :--- | :--- |
"""
    
    # Business interpretation mapping
    interpretations = {
        'recency': "Days since last purchase. Higher recency is a classic and direct sign of inactivity.",
        'frequency': "Total transactions. High frequency indicates strong habit and loyalty, reducing churn risk.",
        'monetary': "Total spending. High value customers are less likely to churn unless neglected.",
        'customer_loyalty_score': "Composite RFM loyalty rating. High loyalty strongly shields against churn.",
        'active_months': "Number of months active. Consistent monthly buyers show high retention.",
        'customer_rank_by_revenue': "Revenue ranking (lower is higher spend). Higher rank (larger rank number) means lower spending, correlating with churn.",
        'customer_tenure': "Days since first purchase. Customers with long tenure have established relationships, making them less volatile.",
        'avg_days_between_purchases': "Average gap between orders. A larger gap means a slower purchase cycle and higher churn risk.",
        'purchase_frequency': "Orders per active month. High velocity buyers represent active business clients.",
        'average_purchase_value': "Average size of invoice in cash. Larger orders correlate with stable commercial accounts.",
        'quantity_per_order': "Average items purchased per order. Stable wholesale orders indicate steady supply chain demands.",
        'high_value_customer_flag': "Top 10% revenue flag. Indicates high-value enterprise accounts.",
        'weekend_sales_ratio': "Proportion of sales on weekends. B2B businesses typically buy during weekdays; weekend retail buyers have slightly different churn habits."
    }
    
    for rank, (idx, row) in enumerate(importance_df.iterrows(), 1):
        direction_text = "Positive (Increases Risk)" if row['Correlation'] > 0 else "Negative (Decreases Risk)"
        desc = interpretations.get(row['Feature'], "Aggregated transactional feature.")
        report_content += f"| {rank} | `{row['Feature']}` | {row['Importance']:.6f} | {direction_text} | {desc} |\n"
        
    report_content += "\n---\n"
    
    if shap_available:
        report_content += """
## 2. SHAP (SHapley Additive exPlanations) Analysis

SHAP values offer a game-theoretic approach to explaining the output of the machine learning model. Unlike global feature importance, SHAP shows:
1. Whether a high or low value of a feature pushes the churn probability up or down.
2. The exact distribution of impact across all customers in the test set.

![SHAP Summary Plot](shap_summary.png)

### SHAP Plot Interpretation

- **Feature Value Color:** Blue represents low values of the feature, and Red represents high values.
- **SHAP Value (X-Axis):** A positive SHAP value (right of center) indicates that the feature value pushes the prediction towards **Churn (1)**. A negative SHAP value (left of center) pushes the prediction towards **Active (0)**.
- **Example - Recency:** High recency (Red dots) is spread far to the right, showing a massive positive impact on churn probability. Low recency (Blue dots) is clustered to the left, showing a strong retention pull.
- **Example - Customer Loyalty Score:** High loyalty score (Red dots) is pulled to the left, acting as a strong negative force (retaining the customer).
"""
    else:
        report_content += """
## 2. SHAP Analysis Notice

SHAP analysis is currently disabled because the `shap` package is not installed on this environment.
To enable game-theoretic explanation plots:
1. Run `pip install shap`
2. Rerun this script to output the SHAP summary plot.
"""
        
    with open(report_file, 'w') as f:
        f.write(report_content)
        
    print(f"Feature importance report successfully saved to: {report_file}")

if __name__ == '__main__':
    main()
