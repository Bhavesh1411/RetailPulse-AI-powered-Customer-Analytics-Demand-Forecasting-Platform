import pandas as pd
import numpy as np
import os
import json

def main():
    print("Starting Inventory Audit...")
    
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    io_dir = os.path.join(base_dir, 'inventory_optimization')
    scripts_dir = os.path.join(io_dir, 'scripts')
    reports_dir = os.path.join(io_dir, 'reports')
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(io_dir, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(io_dir, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(io_dir, 'artifacts'), exist_ok=True)
    
    product_features_file = os.path.join(base_dir, 'processed_data', 'product_features.csv')
    daily_sales_file = os.path.join(base_dir, 'processed_data', 'cleaned_sales_dataset.csv')
    
    if not os.path.exists(product_features_file) or not os.path.exists(daily_sales_file):
        print("Required datasets not found.")
        return
        
    print(f"Loading {product_features_file}")
    product_df = pd.read_csv(product_features_file)
    print(f"Loading {daily_sales_file}")
    sales_df = pd.read_csv(daily_sales_file)
    sales_df['date'] = pd.to_datetime(sales_df['date'], dayfirst=True, format='mixed')
    
    # PHASE 1 & 2: INVENTORY DATA & FEATURE AUDIT
    audit_results = {}
    audit_results['total_products'] = product_df['product_id'].nunique()
    audit_results['date_min'] = sales_df['date'].min().strftime('%Y-%m-%d')
    audit_results['date_max'] = sales_df['date'].max().strftime('%Y-%m-%d')
    total_days = (sales_df['date'].max() - sales_df['date'].min()).days + 1
    audit_results['total_days'] = total_days
    
    audit_results['product_features_columns'] = list(product_df.columns)
    
    # Evaluate demand history (daily demand per product)
    daily_demand = sales_df.groupby(['product_id', 'date'])['quantity_sold'].sum().reset_index()
    # Fill missing dates with 0 for accurate daily average
    # We will compute basic stats per product
    product_stats = sales_df.groupby('product_id').agg(
        total_revenue=('sales_amount', 'sum'),
        total_quantity=('quantity_sold', 'sum'),
        transactions=('transaction_id', 'nunique')
    ).reset_index()
    
    # Calculate average daily demand
    product_stats['avg_daily_demand'] = product_stats['total_quantity'] / total_days
    
    # Calculate standard deviation of daily demand (volatility)
    std_demand = daily_demand.groupby('product_id')['quantity_sold'].std().reset_index()
    std_demand.rename(columns={'quantity_sold': 'demand_std_dev'}, inplace=True)
    # Fillna with 0 for products with only 1 day of sales
    std_demand['demand_std_dev'] = std_demand['demand_std_dev'].fillna(0)
    
    product_stats = product_stats.merge(std_demand, on='product_id', how='left')
    product_stats['demand_std_dev'] = product_stats['demand_std_dev'].fillna(0)
    
    # PHASE 3: ABC ANALYSIS (By Revenue)
    # Class A: Top 80% cumulative revenue
    # Class B: Next 15% cumulative revenue
    # Class C: Bottom 5% cumulative revenue
    
    product_stats = product_stats.sort_values(by='total_revenue', ascending=False)
    total_revenue_all = product_stats['total_revenue'].sum()
    product_stats['revenue_pct'] = product_stats['total_revenue'] / total_revenue_all
    product_stats['cum_revenue_pct'] = product_stats['revenue_pct'].cumsum()
    
    def assign_abc(cum_pct):
        if cum_pct <= 0.80:
            return 'A'
        elif cum_pct <= 0.95:
            return 'B'
        else:
            return 'C'
            
    product_stats['abc_class'] = product_stats['cum_revenue_pct'].apply(assign_abc)
    
    total_quantity_all = product_stats['total_quantity'].sum()
    
    abc_summary = product_stats.groupby('abc_class').agg(
        product_count=('product_id', 'count'),
        total_revenue=('total_revenue', 'sum'),
        total_quantity=('total_quantity', 'sum')
    ).reset_index()
    
    abc_summary['revenue_contribution_pct'] = (abc_summary['total_revenue'] / total_revenue_all) * 100
    abc_summary['quantity_contribution_pct'] = (abc_summary['total_quantity'] / total_quantity_all) * 100
    abc_summary['product_count_pct'] = (abc_summary['product_count'] / len(product_stats)) * 100
    
    audit_results['abc_analysis'] = abc_summary.to_dict(orient='records')
    
    # Determine Missing Features
    required_features = ['avg_daily_demand', 'demand_std_dev', 'lead_time', 'holding_cost', 'ordering_cost']
    available_features = list(product_stats.columns)
    missing_features = [f for f in required_features if f not in available_features]
    
    audit_results['available_features'] = available_features
    audit_results['missing_features'] = missing_features
    
    # Save the audit json
    audit_json_path = os.path.join(reports_dir, 'inventory_audit_results.json')
    with open(audit_json_path, 'w') as f:
        json.dump(audit_results, f, indent=4)
        
    print(f"Inventory audit completed and saved to {audit_json_path}")

if __name__ == '__main__':
    main()
