import pandas as pd
import numpy as np
import os

def main():
    print("Starting Feature Engineering Phase...")
    
    # Paths
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data/cleaned_sales_dataset.csv')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
    
    # 1. Load Cleaned Dataset
    print(f"Loading cleaned dataset from: {input_file}")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
    
    initial_features = list(df.columns)
    print(f"Initial features: {initial_features}")
    
    # ----------------------------------------------------
    # PART 2 – DATE FEATURES (Transactional Level)
    # ----------------------------------------------------
    print("Generating Date Features...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df['date'].dt.dayofweek # 0 is Monday, 6 is Sunday
    df['day_name'] = df['date'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # ----------------------------------------------------
    # PART 3 – SALES FEATURES (Transactional Line Level)
    # ----------------------------------------------------
    print("Generating Sales Features...")
    # sales_per_unit = sales_amount / quantity_sold (handling division by zero just in case)
    df['sales_per_unit'] = np.where(df['quantity_sold'] != 0, df['sales_amount'] / df['quantity_sold'], 0.0)
    
    # Save enhanced transactional dataset
    enhanced_file = os.path.join(output_dir, 'cleaned_sales_dataset_enhanced.csv')
    df.to_csv(enhanced_file, index=False)
    print(f"Enhanced transactional dataset saved to: {enhanced_file}")
    
    # ----------------------------------------------------
    # PART 1 & 5 – CUSTOMER FEATURES & BEHAVIOR (Customer Level)
    # ----------------------------------------------------
    print("Generating Customer (RFM) & Behavioral Features...")
    # Filter for non-null customer_id
    cust_df = df[df['customer_id'].notna()].copy()
    
    # Convert customer_id to integer string format
    cust_df['customer_id'] = cust_df['customer_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Reference date is max date in dataset + 1 day
    reference_date = df['date'].max() + pd.Timedelta(days=1)
    
    # Calculate customer aggregations
    # To properly count unique transactions, we group by customer and use transaction_id
    customer_groups = cust_df.groupby('customer_id')
    
    customer_features = customer_groups.agg(
        last_purchase_date=('date', 'max'),
        first_purchase_date=('date', 'min'),
        total_orders=('transaction_id', 'nunique'),
        monetary=('sales_amount', 'sum'),
        total_quantity=('quantity_sold', 'sum')
    ).reset_index()
    
    # Recency: Days since last purchase
    customer_features['recency'] = (reference_date - customer_features['last_purchase_date']).dt.days
    
    # Frequency: Number of purchases (nunique transaction_id)
    customer_features['frequency'] = customer_features['total_orders']
    
    # Average purchase value (AOV)
    customer_features['average_purchase_value'] = np.where(
        customer_features['total_orders'] > 0,
        customer_features['monetary'] / customer_features['total_orders'],
        0.0
    )
    
    # Quantity per order
    customer_features['quantity_per_order'] = np.where(
        customer_features['total_orders'] > 0,
        customer_features['total_quantity'] / customer_features['total_orders'],
        0.0
    )
    
    # Active Months: unique year-month periods
    # We can compute this by creating a year-month column and counting unique values
    cust_df['year_month'] = cust_df['date'].dt.to_period('M')
    active_months_series = cust_df.groupby('customer_id')['year_month'].nunique()
    customer_features = customer_features.merge(active_months_series, on='customer_id', how='left')
    customer_features.rename(columns={'year_month': 'active_months'}, inplace=True)
    
    # Purchase Frequency: total_orders / active_months
    customer_features['purchase_frequency'] = np.where(
        customer_features['active_months'] > 0,
        customer_features['total_orders'] / customer_features['active_months'],
        0.0
    )
    
    # ----------------------------------------------------
    # ADVANCED CUSTOMER INTELLIGENCE FEATURES
    # ----------------------------------------------------
    print("Generating Advanced Customer Intelligence Features...")
    
    # Rank by revenue descending
    customer_features['customer_rank_by_revenue'] = customer_features['monetary'].rank(ascending=False, method='min').astype(int)
    
    customer_features['customer_tenure'] = (reference_date - customer_features['first_purchase_date']).dt.days
    
    customer_features['avg_days_between_purchases'] = np.where(
        customer_features['total_orders'] > 1,
        customer_features['customer_tenure'] / (customer_features['total_orders'] - 1),
        customer_features['customer_tenure']
    )
    customer_features['avg_days_between_purchases'] = np.where(
        customer_features['avg_days_between_purchases'] == 0,
        30.0,
        customer_features['avg_days_between_purchases']
    )
    
    customer_features['customer_risk_score'] = np.minimum(
        1.0,
        customer_features['recency'] / (2 * customer_features['avg_days_between_purchases'] + 30.0)
    )
    customer_features['churn_warning_flag'] = (customer_features['customer_risk_score'] > 0.7).astype(int)
    
    # Lost 1-Time Buyer Flag: Frequency = 1 and Recency > 90 days
    customer_features['lost_one_time_buyer_flag'] = np.where(
        (customer_features['frequency'] == 1) & (customer_features['recency'] > 90),
        1,
        0
    )
    
    m_threshold = customer_features['monetary'].quantile(0.9)
    customer_features['high_value_customer_flag'] = (customer_features['monetary'] >= m_threshold).astype(int)
    
    # Customer Loyalty Score (1 to 10 scale based on weighted RFM)
    # Recency (lower is better): qcut maps 5 to lowest recency values
    r_score = pd.qcut(customer_features['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(float)
    f_score = pd.qcut(customer_features['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    m_score = pd.qcut(customer_features['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(float)
    composite_score = 0.2 * r_score + 0.3 * f_score + 0.5 * m_score
    customer_features['customer_loyalty_score'] = np.round(1.0 + (composite_score - 1.0) * (9.0 / 4.0), 2)
    
    # Weekend Sales Ratio for customers
    weekend_sales = cust_df[cust_df['is_weekend'] == 1].groupby('customer_id')['sales_amount'].sum().reset_index(name='weekend_sales')
    customer_features = customer_features.merge(weekend_sales, on='customer_id', how='left')
    customer_features['weekend_sales'] = customer_features['weekend_sales'].fillna(0.0)
    customer_features['weekend_sales_ratio'] = np.where(
        customer_features['monetary'] > 0,
        customer_features['weekend_sales'] / customer_features['monetary'],
        0.0
    )
    customer_features['weekend_sales_ratio'] = np.clip(customer_features['weekend_sales_ratio'], 0.0, 1.0)
    customer_features.drop(columns=['weekend_sales'], inplace=True)
    
    # Clean intermediate columns
    customer_features.drop(columns=['last_purchase_date', 'first_purchase_date', 'total_quantity'], inplace=True)
    
    # Reorder columns
    cust_cols = [
        'customer_id', 'recency', 'frequency', 'monetary', 
        'average_purchase_value',
        'quantity_per_order', 'active_months', 'purchase_frequency',
        'customer_loyalty_score', 'customer_risk_score',
        'high_value_customer_flag', 'customer_tenure', 'avg_days_between_purchases',
        'weekend_sales_ratio', 'customer_rank_by_revenue',
        'churn_warning_flag', 'lost_one_time_buyer_flag'
    ]
    customer_features = customer_features[cust_cols]
    
    customer_file = os.path.join(output_dir, 'customer_features.csv')
    customer_features.to_csv(customer_file, index=False)
    print(f"Customer features dataset saved to: {customer_file}")
    
    # ----------------------------------------------------
    # PART 6 – PRODUCT & INVENTORY FEATURES (Product Level)
    # ----------------------------------------------------
    print("Generating Product & Inventory Features...")
    product_groups = df.groupby('product_id')
    
    product_features = product_groups.agg(
        total_product_sales=('sales_amount', 'sum'),
        total_product_quantity=('quantity_sold', 'sum'),
        average_product_revenue=('sales_amount', 'mean')
    ).reset_index()
    
    product_features['product_velocity'] = product_features['total_product_quantity'] / 376.0
    product_features['product_total_revenue'] = product_features['total_product_sales']
    
    # Ranks
    product_features['product_rank_by_revenue'] = product_features['total_product_sales'].rank(ascending=False, method='min').astype(int)
    product_features['product_popularity_rank'] = product_features['total_product_quantity'].rank(ascending=False, method='min').astype(int)
    
    # Revenue contribution
    grand_total_revenue = df['sales_amount'].sum()
    product_features['product_revenue_contribution'] = product_features['product_total_revenue'] / (grand_total_revenue + 1e-5)
    
    # Weekend Sales Ratio for products (based on quantities)
    weekend_qty = df[df['is_weekend'] == 1].groupby('product_id')['quantity_sold'].sum().reset_index(name='weekend_qty')
    product_features = product_features.merge(weekend_qty, on='product_id', how='left')
    product_features['weekend_qty'] = product_features['weekend_qty'].fillna(0.0)
    product_features['weekend_sales_ratio'] = np.where(
        product_features['total_product_quantity'] > 0,
        product_features['weekend_qty'] / product_features['total_product_quantity'],
        0.0
    )
    product_features['weekend_sales_ratio'] = np.clip(product_features['weekend_sales_ratio'], 0.0, 1.0)
    product_features.drop(columns=['weekend_qty'], inplace=True)
    
    # Slow moving inventory flag
    last_sale = df.groupby('product_id')['date'].max().reset_index(name='last_sale_date')
    product_features = product_features.merge(last_sale, on='product_id', how='left')
    product_features['days_since_last_sale'] = (reference_date - product_features['last_sale_date']).dt.days
    vel_threshold = product_features['product_velocity'].quantile(0.25)
    product_features['slow_moving_inventory_flag'] = np.where(
        (product_features['product_velocity'] <= vel_threshold) & (product_features['days_since_last_sale'] > 30),
        1,
        0
    )
    product_features.drop(columns=['last_sale_date', 'days_since_last_sale'], inplace=True)
    
    # Daily demand std over 376 days using exact mathematical variance formula
    qty_sq = df.copy()
    qty_sq['qty_sq'] = qty_sq['quantity_sold'] ** 2
    qty_sq_grouped = qty_sq.groupby('product_id')['qty_sq'].sum().reset_index()
    product_features = product_features.merge(qty_sq_grouped, on='product_id', how='left')
    
    s = product_features['total_product_quantity']
    ss = product_features['qty_sq']
    var = np.maximum(0.0, (ss - (s ** 2) / 376.0) / 375.0)
    product_features['daily_demand_std'] = np.sqrt(var)
    product_features.drop(columns=['qty_sq'], inplace=True)
    
    # Inventory metrics
    # Protect against products with negative net velocity (products with net returns)
    inv_velocity = np.maximum(0.0, product_features['product_velocity'])
    
    lead_time = 7.0
    product_features['safety_stock'] = 1.96 * product_features['daily_demand_std'] * np.sqrt(lead_time)
    product_features['reorder_point'] = inv_velocity * lead_time + product_features['safety_stock']
    product_features['assumed_current_stock'] = inv_velocity * 10.0
    
    # Days to stockout: 999.0 if no demand
    product_features['days_to_stockout'] = np.where(
        inv_velocity > 0,
        product_features['assumed_current_stock'] / inv_velocity,
        999.0
    )
    
    # Low stock flag: 0 if no demand
    product_features['low_stock_flag'] = np.where(
        inv_velocity > 0,
        (product_features['assumed_current_stock'] < product_features['reorder_point']).astype(int),
        0
    )
    
    # Inventory risk score: 0.0 if no demand
    product_features['inventory_risk_score'] = np.where(
        inv_velocity > 0,
        np.minimum(1.0, product_features['reorder_point'] / (product_features['assumed_current_stock'] + 1e-5)),
        0.0
    )
    product_features['inventory_risk_score'] = np.clip(product_features['inventory_risk_score'], 0.0, 1.0)
    
    # Save product dataset
    product_file = os.path.join(output_dir, 'product_features.csv')
    product_features.to_csv(product_file, index=False)
    print(f"Product features dataset saved to: {product_file}")
    
    # ----------------------------------------------------
    # PART 4 – TIME SERIES FEATURES & SEASONALITY (Daily Forecasting Level)
    # ----------------------------------------------------
    print("Generating Daily Time Series Forecasting & Seasonality Features...")
    # Aggregate daily
    df['day_date'] = df['date'].dt.normalize()
    daily_df = df.groupby('day_date').agg(
        sales_amount=('sales_amount', 'sum'),
        quantity_sold=('quantity_sold', 'sum')
    ).reset_index()
    
    # Set day_date as index and reindex to fill any missing days with 0
    daily_df.set_index('day_date', inplace=True)
    full_date_range = pd.date_range(start=daily_df.index.min(), end=daily_df.index.max(), freq='D')
    daily_df = daily_df.reindex(full_date_range, fill_value=0.0).reset_index()
    daily_df.rename(columns={'index': 'date'}, inplace=True)
    
    # Rolling 7-day and 30-day averages
    daily_df['rolling_7_day_sales'] = daily_df['sales_amount'].rolling(window=7, min_periods=1).mean()
    daily_df['rolling_30_day_sales'] = daily_df['sales_amount'].rolling(window=30, min_periods=1).mean()
    daily_df['rolling_7_day_quantity'] = daily_df['quantity_sold'].rolling(window=7, min_periods=1).mean()
    daily_df['rolling_30_day_quantity'] = daily_df['quantity_sold'].rolling(window=30, min_periods=1).mean()
    
    # Lags (1, 7, 30 days)
    daily_df['lag_1_sales'] = daily_df['sales_amount'].shift(1)
    daily_df['lag_7_sales'] = daily_df['sales_amount'].shift(7)
    daily_df['lag_30_sales'] = daily_df['sales_amount'].shift(30)
    
    daily_df['lag_1_qty'] = daily_df['quantity_sold'].shift(1)
    daily_df['lag_7_qty'] = daily_df['quantity_sold'].shift(7)
    daily_df['lag_30_qty'] = daily_df['quantity_sold'].shift(30)
    
    # Weekday Seasonality Index
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    overall_mean_sales = daily_df['sales_amount'].mean()
    weekday_means = daily_df.groupby('day_of_week')['sales_amount'].mean().reset_index(name='weekday_mean_sales')
    weekday_means['weekday_seasonality_index'] = weekday_means['weekday_mean_sales'] / (overall_mean_sales + 1e-5)
    daily_df = daily_df.merge(weekday_means[['day_of_week', 'weekday_seasonality_index']], on='day_of_week', how='left')
    daily_df.drop(columns=['day_of_week'], inplace=True)
    
    # Monthly Seasonality Index
    daily_df['month'] = daily_df['date'].dt.month
    monthly_means = daily_df.groupby('month')['sales_amount'].mean().reset_index(name='monthly_mean_sales')
    monthly_means['monthly_seasonality_index'] = monthly_means['monthly_mean_sales'] / (overall_mean_sales + 1e-5)
    daily_df = daily_df.merge(monthly_means[['month', 'monthly_seasonality_index']], on='month', how='left')
    daily_df.drop(columns=['month'], inplace=True)
    
    # Composite seasonality index (removed – redundant with weekday and monthly indices)

    # 30-day sales volatility
    daily_df['sales_volatility_30_day'] = daily_df['sales_amount'].rolling(window=30, min_periods=1).std().fillna(0.0)

    # Demand Momentum (7-day sales / 30-day sales)
    daily_df['demand_momentum'] = np.where(
        daily_df['rolling_30_day_sales'] > 0,
        daily_df['rolling_7_day_sales'] / daily_df['rolling_30_day_sales'],
        1.0
    )

    # Quantity Momentum (7-day qty / 30-day qty)
    daily_df['quantity_momentum'] = np.where(
        daily_df['rolling_30_day_quantity'] > 0,
        daily_df['rolling_7_day_quantity'] / daily_df['rolling_30_day_quantity'],
        1.0
    )

    # MoM Sales Growth Rate (removed to avoid leakage – uses future data)
    
    # Save daily time series forecasting dataset
    daily_file = os.path.join(output_dir, 'daily_sales_forecast_features.csv')
    daily_df.to_csv(daily_file, index=False)
    print(f"Daily forecasting features dataset saved to: {daily_file}")
    
    # ----------------------------------------------------
    # PART 7 – FEATURE VALIDATION
    # ----------------------------------------------------
    print("Performing Feature Validation...")
    
    cust_val_cols = [
        'recency', 'frequency', 'monetary', 'average_purchase_value', 
        'quantity_per_order', 'active_months', 'purchase_frequency',
        'customer_loyalty_score', 'customer_risk_score',
        'high_value_customer_flag', 'customer_tenure', 'avg_days_between_purchases',
        'weekend_sales_ratio', 'customer_rank_by_revenue',
        'churn_warning_flag', 'lost_one_time_buyer_flag'
    ]
    
    prod_val_cols = [
        'total_product_sales', 'total_product_quantity', 'average_product_revenue',
        'product_popularity_rank', 'product_velocity', 'product_total_revenue', 
        'product_rank_by_revenue', 'product_revenue_contribution', 'weekend_sales_ratio', 
        'slow_moving_inventory_flag', 'daily_demand_std', 'safety_stock', 'reorder_point', 
        'assumed_current_stock', 'days_to_stockout', 'low_stock_flag', 'inventory_risk_score'
    ]
    
    daily_val_cols = [
        'sales_amount', 'quantity_sold', 'rolling_7_day_sales', 'rolling_30_day_sales',
        'rolling_7_day_quantity', 'rolling_30_day_quantity', 'lag_1_sales', 'lag_7_sales',
        'lag_30_sales', 'lag_1_qty', 'lag_7_qty', 'lag_30_qty', 'weekday_seasonality_index',
        'monthly_seasonality_index', 'sales_volatility_30_day', 'demand_momentum',
        'quantity_momentum'
    ]
    
    datasets_to_validate = {
        'cleaned_sales_dataset_enhanced.csv': (df, ['year', 'month', 'quarter', 'week_of_year', 'day_of_week', 'is_weekend', 'sales_per_unit']),
        'customer_features.csv': (customer_features, cust_val_cols),
        'product_features.csv': (product_features, prod_val_cols),
        'daily_sales_forecast_features.csv': (daily_df, daily_val_cols)
    }
    
    validation_results = {}
    for name, (data, cols) in datasets_to_validate.items():
        res = {}
        for col in cols:
            nulls = int(data[col].isna().sum())
            infs = int(np.isinf(data[col]).sum()) if np.issubdtype(data[col].dtype, np.number) else 0
            stats_dict = {
                'nulls': nulls,
                'infs': infs,
                'min': float(data[col].min()) if np.issubdtype(data[col].dtype, np.number) else str(data[col].min()),
                'max': float(data[col].max()) if np.issubdtype(data[col].dtype, np.number) else str(data[col].max()),
                'mean': float(data[col].mean()) if np.issubdtype(data[col].dtype, np.number) else None
            }
            res[col] = stats_dict
        validation_results[name] = res
        
    validation_report_file = os.path.join(output_dir, 'feature_validation_summary.json')
    import json
    with open(validation_report_file, 'w') as f:
        json.dump(validation_results, f, indent=4)
        
    print(f"Validation summary saved to: {validation_report_file}")
    print("Feature Engineering completed successfully.")

if __name__ == '__main__':
    main()
