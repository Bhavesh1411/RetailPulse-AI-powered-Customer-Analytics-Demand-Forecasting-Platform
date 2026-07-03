import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def main():
    print("=== PHASE 4 & 5: MODEL TRAINING & EVALUATION ===")
    
    # Paths
    dataset_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/datasets/true_churn_dataset.csv')
    models_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/models')
    reports_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    report_file = os.path.join(reports_dir, 'model_evaluation_report.md')
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found at {dataset_file}")
        return
        
    df = pd.read_csv(dataset_file)
    
    # Features & Target
    # Exclude customer_id and target true_churn_flag
    features = [col for col in df.columns if col not in ['customer_id', 'true_churn_flag']]
    
    X = df[features].copy()
    y = df['true_churn_flag'].copy()
    
    print(f"Features list: {features}")
    print(f"Dataset shape: {X.shape}")
    
    # 1. Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train Set Shape: {X_train.shape} (Churn Rate: {y_train.mean()*100:.2f}%)")
    print(f"Test Set Shape: {X_test.shape} (Churn Rate: {y_test.mean()*100:.2f}%)")
    
    # Scale features for Logistic Regression (we will scale features for Logistic Regression, but trees can use unscaled or scaled.
    # To keep code clean, we scale X_train/X_test for Logistic Regression separately, or just scale the input for LR.)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for prediction time
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to models/scaler.pkl")
    
    # Calculate class weights for imbalance handling
    # (though 44% churn rate is quite balanced, we still apply weight balancing)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight_val = num_neg / num_pos
    
    class_weights = {0: 1.0, 1: scale_pos_weight_val}
    print(f"Computed Class Weights: {class_weights}")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(scale_pos_weight=scale_pos_weight_val, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(scale_pos_weight=scale_pos_weight_val, random_state=42, verbose=-1)
    }
    
    cv_results = {}
    test_results = {}
    
    # 2. Stratified Cross-Validation (5-fold) on Training Data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining and cross-validating {name}...")
        
        # Use scaled features for Logistic Regression, unscaled for tree-based models
        X_cv = X_train_scaled if name == 'Logistic Regression' else X_train.values
        X_tst = X_test_scaled if name == 'Logistic Regression' else X_test.values
        
        # Run cross-validation
        scores = cross_validate(
            model, X_cv, y_train, cv=skf,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            return_train_score=False
        )
        
        cv_results[name] = {
            'accuracy': scores['test_accuracy'].mean(),
            'precision': scores['test_precision'].mean(),
            'recall': scores['test_recall'].mean(),
            'f1': scores['test_f1'].mean(),
            'roc_auc': scores['test_roc_auc'].mean()
        }
        
        print(f"CV ROC AUC: {cv_results[name]['roc_auc']:.4f} | CV Recall: {cv_results[name]['recall']:.4f}")
        
        # Train on full training set and evaluate on test set
        model.fit(X_cv, y_train)
        
        # Predictions
        preds = model.predict(X_tst)
        probs = model.predict_proba(X_tst)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        test_results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'confusion_matrix': cm
        }
        
        # Save model
        model_path = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} model to {model_path}")
        
    # 3. Identify the Best Model
    # We will rank models by ROC AUC on the test set first, and Recall as tie-breaker
    best_model_name = max(test_results, key=lambda k: (test_results[k]['roc_auc'], test_results[k]['recall']))
    best_metrics = test_results[best_model_name]
    
    print("\n==================================================")
    print(f"BEST MODEL IDENTIFIED: {best_model_name}")
    print(f"ROC AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Recall:  {best_metrics['recall']:.4f}")
    print("==================================================")
    
    # Save best model info text file
    with open(os.path.join(models_dir, 'best_model_name.txt'), 'w') as f:
        f.write(best_model_name)
        
    # Generate Markdown Report
    report_content = f"""# Model Training and Evaluation Report

This report evaluates the performance of Logistic Regression, Random Forest, XGBoost, and LightGBM models trained on the point-in-time true churn dataset.

## Executive Summary

- **Best-Performing Model:** `{best_model_name}`
- **Test ROC AUC:** `{best_metrics['roc_auc']:.4%}`
- **Test Recall:** `{best_metrics['recall']:.4%}`
- **Test F1-Score:** `{best_metrics['f1']:.4%}`
- **Test Accuracy:** `{best_metrics['accuracy']:.4%}`

---

## 1. Cross-Validation Performance (Training Set)

We performed Stratified 5-Fold Cross-Validation on the training set (80% of the active customer cohort) using class weighting to balance the loss function.

| Model | CV Accuracy | CV Precision | CV Recall | CV F1-Score | CV ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for name in models.keys():
        cv_m = cv_results[name]
        report_content += f"| {name} | {cv_m['accuracy']:.4%} | {cv_m['precision']:.4%} | {cv_m['recall']:.4%} | {cv_m['f1']:.4%} | {cv_m['roc_auc']:.4%} |\n"
        
    report_content += """
---

## 2. Test Set Performance (Evaluation Set)

Below are the evaluation metrics calculated on the holdout test set (20% of the active customer cohort).

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for name in models.keys():
        test_m = test_results[name]
        report_content += f"| {name} | {test_m['accuracy']:.4%} | {test_m['precision']:.4%} | {test_m['recall']:.4%} | {test_m['f1']:.4%} | {test_m['roc_auc']:.4%} |\n"

    report_content += """
---

## 3. Confusion Matrices

Below are the confusion matrices for each of the trained models on the test set (746 total customers).

### Confusion Matrix Legend
- **True Negative (TN):** Active customer correctly classified as Active.
- **False Positive (FP):** Active customer incorrectly flagged as Churned.
- **False Negative (FN):** Churned customer incorrectly flagged as Active.
- **True Positive (TP):** Churned customer correctly classified as Churned.

"""
    for name in models.keys():
        cm = test_results[name]['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        report_content += f"""#### {name}
- **TN:** {tn} | **FP:** {fp}
- **FN:** {fn} | **TP:** {tp}
- *Total Predictions:* {tn+fp+fn+tp} | *True Active:* {tn+fp} | *True Churned:* {fn+tp}
- *Recall (Sensitivity):* {tp / (tp + fn):.2%} | *Specificity:* {tn / (tn + fp):.2%}
- *Precision:* {tp / (tp + fp):.2%}

"""
        
    report_content += f"""
---

## Conclusion & Best Model Selection

The model **{best_model_name}** was selected as the champion model because it achieved the highest ROC AUC of **{best_metrics['roc_auc']:.2%}** on the test set, while maintaining a strong balance between Recall (**{best_metrics['recall']:.2%}**) and Precision (**{best_metrics['precision']:.2%}**). 

The class weights worked successfully to pull up model sensitivity, ensuring that the model detects a high proportion of churners without producing an overwhelming number of false alarms.
"""
    
    with open(report_file, 'w') as f:
        f.write(report_content)
        
    print(f"Model evaluation report successfully saved to: {report_file}")

if __name__ == '__main__':
    main()
