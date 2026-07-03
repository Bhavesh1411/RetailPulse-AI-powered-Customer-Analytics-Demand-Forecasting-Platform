import pandas as pd
import numpy as np
import os

input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data/customer_features.csv')
output_cleaned_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data/customer_features_cleaned.csv')
output_report_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data/customer_features_cleaning_report.csv')

print(f"Loading {input_file} for cleaning...")
df = pd.read_csv(input_file)

report_data = []

def record_report(col, orig_dtype, final_dtype, nulls_before, nulls_after, corrections):
    report_data.append({
        'Column': col,
        'Original_DataType': orig_dtype,
        'Final_DataType': final_dtype,
        'Nulls_Before': nulls_before,
        'Nulls_After': nulls_after,
        'Corrections_Performed': "; ".join(corrections) if corrections else "None"
    })

for col in df.columns:
    orig_dtype = str(df[col].dtype)
    nulls_before = df[col].isna().sum()
    corrections = []
    
    # 1. Handle IDs
    if col == 'customer_id':
        # Ensure string, replace .0, drop duplicates
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(r'\.0$', '', regex=True)
        corrections.append("Converted to string; Stripped trailing .0")
        
        dups = df[col].duplicated().sum()
        if dups > 0:
            df = df.drop_duplicates(subset=[col], keep='first')
            corrections.append(f"Removed {dups} duplicate IDs")

    # 2. Handle Categorical / Strings
    elif col in ['gender', 'city', 'state', 'customer_name']:
        df[col] = df[col].astype(str).str.strip().str.title()
        # replace 'Nan' or 'None' strings back to actual NaN if needed
        df[col] = df[col].replace(['Nan', 'None'], np.nan)
        corrections.append("Standardized to Title Case; Stripped whitespace")
        
    # 3. Handle Numeric Integrals
    elif col in ['recency', 'frequency', 'total_orders', 'active_months', 
                 'customer_rank_by_revenue', 'churn_warning_flag', 
                 'high_value_customer_flag', 'customer_tenure', 'lost_one_time_buyer_flag']:
        # check for inf and impossibles
        infs = np.isinf(df[col]).sum()
        if infs > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            corrections.append(f"Replaced {infs} infinite values with NaN")
            
        # check for negatives (except if rank could be? No.)
        negs = (df[col] < 0).sum()
        if negs > 0:
            # Business logic: recency, frequency, tenure shouldn't be negative
            df[col] = df[col].clip(lower=0)
            corrections.append(f"Clipped {negs} negative values to 0")
            
        # Convert to Integer type
        try:
            # fillna with 0 or a placeholder before converting to int? No, use Int64 (nullable int)
            df[col] = df[col].astype('Int64')
            corrections.append("Converted float to nullable Integer (Int64)")
        except Exception as e:
            corrections.append(f"Failed to convert to Int64: {str(e)}")

    # 4. Handle Numeric Floats
    elif pd.api.types.is_numeric_dtype(df[col]):
        infs = np.isinf(df[col]).sum()
        if infs > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            corrections.append(f"Replaced {infs} infinite values with NaN")
            
        if col in ['monetary', 'average_purchase_value', 'customer_lifetime_value', 
                   'customer_loyalty_score', 'customer_risk_score', 'purchase_frequency', 
                   'avg_days_between_purchases', 'weekend_sales_ratio']:
            negs = (df[col] < 0).sum()
            if negs > 0:
                df[col] = df[col].clip(lower=0.0)
                corrections.append(f"Clipped {negs} negative values to 0.0")

    final_dtype = str(df[col].dtype)
    nulls_after = df[col].isna().sum()
    record_report(col, orig_dtype, final_dtype, nulls_before, nulls_after, corrections)

# Save Cleaned CSV
try:
    df.to_csv(output_cleaned_file, index=False)
    print(f"Cleaned dataset saved to: {output_cleaned_file}")
except PermissionError:
    print("\n" + "="*80)
    print(f"ERROR: Permission denied when writing to:\n{output_cleaned_file}")
    print("This usually means the file is open in Microsoft Excel or another program.")
    print("Please close the file in Excel or other applications, then run this script again.")
    print("="*80 + "\n")
    # Save a copy as fallback
    fallback_file = output_cleaned_file.replace('.csv', '_fallback.csv')
    df.to_csv(fallback_file, index=False)
    print(f"Saved fallback copy to: {fallback_file}")

# Save Report CSV
try:
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_report_file, index=False)
    print(f"Cleaning report saved to: {output_report_file}")
except PermissionError:
    print(f"ERROR: Permission denied when writing report to:\n{output_report_file}")
    fallback_report = output_report_file.replace('.csv', '_fallback.csv')
    report_df.to_csv(fallback_report, index=False)
    print(f"Saved fallback report to: {fallback_report}")

