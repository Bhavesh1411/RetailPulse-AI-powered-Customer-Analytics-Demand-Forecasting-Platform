import pandas as pd
import numpy as np
import os

def main():
    print("=== PHASE 1: POINT-IN-TIME DATASET CREATION ===")
    
    # Paths
    input_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'processed_data/cleaned_sales_dataset.csv')
    output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'churn_prediction_true/datasets')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'true_churn_dataset.csv')
    
    print(f"Loading raw transaction dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
    
    # Clean customer IDs (remove nulls, convert to string, format)
    df = df[df['customer_id'].notna()].copy()
    df['customer_id'] = df['customer_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    cutoff_date = pd.to_datetime('2010-09-10')
    pred_start_date = pd.to_datetime('2010-09-11')
    pred_end_date = pd.to_datetime('2010-12-09')
    
    print(f"Cutoff Date (T): {cutoff_date.date()}")
    print(f"Observation Window: {df['date'].min().date()} to {cutoff_date.date()}")
    print(f"Prediction Window: {pred_start_date.date()} to {pred_end_date.date()}")
    
    # Split transactions based on cutoff
    obs_df = df[df['date'] <= cutoff_date].copy()
    pred_df = df[(df['date'] >= pred_start_date) & (df['date'] <= pred_end_date)].copy()
    
    # Define active cohort in observation window
    obs_customers = obs_df['customer_id'].unique()
    print(f"Number of unique customers active in observation window: {len(obs_customers)}")
    
    # reference date for recency and tenure
    reference_date = cutoff_date + pd.Timedelta(days=1)
    
    # Date calculations on transaction level
    obs_df['day_of_week'] = obs_df['date'].dt.dayofweek
    obs_df['is_weekend'] = obs_df['day_of_week'].isin([5, 6]).astype(int)
    
    print("Computing customer-level features...")
    # Group by customer
    groups = obs_df.groupby('customer_id')
    
    # Aggregate features
    cust_features = groups.agg(
        last_purchase_date=('date', 'max'),
        first_purchase_date=('date', 'min'),
        frequency=('transaction_id', 'nunique'),
        monetary=('sales_amount', 'sum'),
        total_quantity=('quantity_sold', 'sum')
    ).reset_index()
    
    # Recency: Days between last purchase and reference date
    cust_features['recency'] = (reference_date - cust_features['last_purchase_date']).dt.days
    
    # Average Purchase Value (AOV)
    cust_features['average_purchase_value'] = np.where(
        cust_features['frequency'] > 0,
        cust_features['monetary'] / cust_features['frequency'],
        0.0
    )
    
    # Customer Tenure
    cust_features['customer_tenure'] = (reference_date - cust_features['first_purchase_date']).dt.days
    
    # Active Months
    obs_df['year_month'] = obs_df['date'].dt.to_period('M')
    active_months_series = obs_df.groupby('customer_id')['year_month'].nunique().reset_index()
    active_months_series.rename(columns={'year_month': 'active_months'}, inplace=True)
    cust_features = cust_features.merge(active_months_series, on='customer_id', how='left')
    
    # Purchase Frequency: frequency / active_months
    cust_features['purchase_frequency'] = np.where(
        cust_features['active_months'] > 0,
        cust_features['frequency'] / cust_features['active_months'],
        0.0
    )
    
    # Avg Days Between Purchases
    cust_features['avg_days_between_purchases'] = np.where(
        cust_features['frequency'] > 1,
        cust_features['customer_tenure'] / (cust_features['frequency'] - 1),
        cust_features['customer_tenure']
    )
    cust_features['avg_days_between_purchases'] = np.where(
        cust_features['avg_days_between_purchases'] == 0,
        30.0,
        cust_features['avg_days_between_purchases']
    )
    
    # Weekend Sales Ratio
    weekend_sales = obs_df[obs_df['is_weekend'] == 1].groupby('customer_id')['sales_amount'].sum().reset_index(name='weekend_sales')
    cust_features = cust_features.merge(weekend_sales, on='customer_id', how='left')
    cust_features['weekend_sales'] = cust_features['weekend_sales'].fillna(0.0)
    cust_features['weekend_sales_ratio'] = np.where(
        cust_features['monetary'] > 0,
        cust_features['weekend_sales'] / cust_features['monetary'],
        0.0
    )
    cust_features['weekend_sales_ratio'] = np.clip(cust_features['weekend_sales_ratio'], 0.0, 1.0)
    cust_features.drop(columns=['weekend_sales'], inplace=True)
    
    # Customer Loyalty Score (1 to 10 scale based on weighted RFM)
    r_score = pd.qcut(cust_features['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(float)
    f_score = pd.qcut(cust_features['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    m_score = pd.qcut(cust_features['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    composite_score = 0.2 * r_score + 0.3 * f_score + 0.5 * m_score
    cust_features['customer_loyalty_score'] = np.round(1.0 + (composite_score - 1.0) * (9.0 / 4.0), 2)
    
    # Quantity Per Order
    cust_features['quantity_per_order'] = np.where(
        cust_features['frequency'] > 0,
        cust_features['total_quantity'] / cust_features['frequency'],
        0.0
    )
    
    # Customer Rank by Revenue
    cust_features['customer_rank_by_revenue'] = cust_features['monetary'].rank(ascending=False, method='min').astype(int)
    
    # High Value Customer Flag (top 10% of revenue)
    m_threshold = cust_features['monetary'].quantile(0.9)
    cust_features['high_value_customer_flag'] = (cust_features['monetary'] >= m_threshold).astype(int)
    
    # ----------------------------------------------------
    # PHASE 2: TARGET CONSTRUCTION (TRUE CHURN LABEL)
    # ----------------------------------------------------
    print("Constructing target variable (true_churn_flag)...")
    
    # Get set of customer_ids active in prediction window
    pred_active_customers = set(pred_df['customer_id'].unique())
    
    # Churned customer = customer not active in prediction window
    # Active customer = customer active in prediction window
    cust_features['true_churn_flag'] = np.where(
        cust_features['customer_id'].isin(pred_active_customers),
        0,
        1
    )
    
    # Drop intermediate columns
    cust_features.drop(columns=['last_purchase_date', 'first_purchase_date', 'total_quantity'], inplace=True)
    
    # Check shape & distribution
    print(f"Calculated feature dataset shape: {cust_features.shape}")
    class_counts = cust_features['true_churn_flag'].value_counts()
    class_pct = cust_features['true_churn_flag'].value_counts(normalize=True) * 100
    
    print("\n--- TARGET VARIABLE CLASS DISTRIBUTION ---")
    for val in [0, 1]:
        label = "Active (0)" if val == 0 else "Churned (1)"
        print(f"{label}: {class_counts.get(val, 0)} customers ({class_pct.get(val, 0.0):.2f}%)")
    
    # Save to file
    cust_features.to_csv(output_file, index=False)
    print(f"Point-in-time dataset successfully saved to: {output_file}")

if __name__ == '__main__':
    main()
