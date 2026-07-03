import pandas as pd
import numpy as np
import os
import pickle

def main():
    print("=== PHASE 7: CUSTOMER RISK SCORING ===")
    
    # Paths
    models_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/models')
    dataset_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/datasets/true_churn_dataset.csv')
    predictions_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    output_file = os.path.join(predictions_dir, 'customer_true_churn_predictions.csv')
    
    best_model_name_path = os.path.join(models_dir, 'best_model_name.txt')
    if not os.path.exists(best_model_name_path):
        print("Error: best_model_name.txt not found. Run training script first.")
        return
        
    with open(best_model_name_path, 'r') as f:
        best_model_name = f.read().strip()
        
    print(f"Best model identified: {best_model_name}")
    
    # Load model
    model_file = os.path.join(models_dir, f"{best_model_name.lower().replace(' ', '_')}_model.pkl")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
        
    # Load data
    df = pd.read_csv(dataset_file)
    features = [col for col in df.columns if col not in ['customer_id', 'true_churn_flag']]
    X = df[features]
    
    # Handle scaling for Logistic Regression
    if best_model_name == 'Logistic Regression':
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_input = scaler.transform(X)
    else:
        X_input = X.values
        
    # Generate probabilities
    print("Generating churn probabilities...")
    probs = model.predict_proba(X_input)[:, 1]
    
    # Create prediction output dataframe
    pred_df = pd.DataFrame({
        'customer_id': df['customer_id'],
        'churn_probability': probs,
        'true_churn_flag': df['true_churn_flag'],
        'recency': df['recency'],
        'frequency': df['frequency'],
        'monetary': df['monetary'],
        'customer_loyalty_score': df['customer_loyalty_score']
    })
    
    # Define risk category thresholds
    # Low Risk: < 30%
    # Medium Risk: 30% <= prob < 70%
    # High Risk: >= 70%
    pred_df['risk_category'] = np.where(
        pred_df['churn_probability'] < 0.30,
        'Low Risk',
        np.where(
            pred_df['churn_probability'] < 0.70,
            'Medium Risk',
            'High Risk'
        )
    )
    
    # Calculate Risk Distribution
    risk_counts = pred_df['risk_category'].value_counts()
    risk_pcts = pred_df['risk_category'].value_counts(normalize=True) * 100
    
    print("\n--- CUSTOMER RISK CATEGORY DISTRIBUTION ---")
    for category in ['Low Risk', 'Medium Risk', 'High Risk']:
        count = risk_counts.get(category, 0)
        pct = risk_pcts.get(category, 0.0)
        print(f"{category}: {count} customers ({pct:.2f}%)")
        
    # Average metrics per risk category
    print("\n--- AVERAGE METRICS BY RISK CATEGORY ---")
    summary = pred_df.groupby('risk_category').agg(
        avg_probability=('churn_probability', 'mean'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean'),
        avg_loyalty=('customer_loyalty_score', 'mean'),
        actual_churn_rate=('true_churn_flag', 'mean')
    ).reindex(['Low Risk', 'Medium Risk', 'High Risk'])
    
    print(summary)
    
    # Save output to predictions
    pred_df.to_csv(output_file, index=False)
    print(f"\nPredictions successfully saved to: {output_file}")

if __name__ == '__main__':
    main()
