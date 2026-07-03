import pandas as pd
import numpy as np
import os
import json

def main():
    print("Starting Day 2 Data Cleaning...")
    
    # 1. Load Dataset
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data/sales_dataset_online_retail.csv')
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # Track stats
    stats = {
        'initial_rows': initial_shape[0],
        'initial_cols': initial_shape[1],
        'duplicates_removed': 0,
        'bad_debt_removed': 0,
        'zero_price_removed': 0,
        'test_bank_charges_removed': 0
    }
    
    # 2. Remove Schema Placeholders
    empty_cols = ['store_id', 'category', 'discount', 'payment_method']
    df.drop(columns=empty_cols, inplace=True, errors='ignore')
    print(f"Dropped empty columns: {empty_cols}")
    
    # 3. Data Type Correction & Validation
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    string_cols = ['transaction_id', 'product_id', 'region']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Standardize customer_id
    if 'customer_id' in df.columns:
        # We need to treat nulls carefully
        # Cast to string, remove .0, replace 'nan' string with np.nan
        df['customer_id'] = df['customer_id'].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', np.nan)
        
    # 4. Remove Duplicates
    dup_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    stats['duplicates_removed'] = int(dup_count)
    print(f"Removed {dup_count} exact duplicates.")
    
    # 5. Treat Data Quality Outliers
    
    # a. Remove Bad Debt adjustments
    # Product ID B with negative price
    if 'product_id' in df.columns and 'unit_price' in df.columns:
        bad_debt_mask = (df['product_id'] == 'B') & (df['unit_price'] < 0)
        stats['bad_debt_removed'] = int(bad_debt_mask.sum())
        df = df[~bad_debt_mask]
        print(f"Removed {stats['bad_debt_removed']} Bad Debt adjustment records.")
        
    # b. Remove zero-price records (write-offs and reconciliations)
    if 'unit_price' in df.columns:
        zero_price_mask = (df['unit_price'] == 0)
        stats['zero_price_removed'] = int(zero_price_mask.sum())
        df = df[~zero_price_mask]
        print(f"Removed {stats['zero_price_removed']} zero-price records.")
        
    # c. Remove test transactions and bank charges
    if 'product_id' in df.columns:
        test_bank_mask = df['product_id'].isin(['TEST001', 'BANK CHARGES'])
        stats['test_bank_charges_removed'] = int(test_bank_mask.sum())
        df = df[~test_bank_mask]
        print(f"Removed {stats['test_bank_charges_removed']} test and bank charge records.")
        
    # 6. Preserve Business Outliers
    # Returns (transactions starting with C and negative quantity) are kept.
    # Bulk orders are kept.
    # No explicit action needed as they are kept by default.
    
    final_shape = df.shape
    stats['final_rows'] = final_shape[0]
    stats['final_cols'] = final_shape[1]
    stats['final_columns'] = list(df.columns)
    print(f"Final shape: {final_shape}")
    
    # 7. Write Cleaned Dataset
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
    output_file = os.path.join(output_dir, 'cleaned_sales_dataset.csv')
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    
    # 8. Generate Reports
    json_report_file = os.path.join(output_dir, 'cleaning_summary_report.json')
    with open(json_report_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Cleaning summary saved to: {json_report_file}")
    
    # Markdown report will be generated via an artifact call
    # writing the markdown file locally too just in case
    md_content = f"""# Data Quality Report

## Cleaning Process Summary
- **Initial rows**: {stats['initial_rows']}
- **Duplicates removed**: {stats['duplicates_removed']}
- **Bad Debt records removed**: {stats['bad_debt_removed']}
- **Zero-price records removed**: {stats['zero_price_removed']}
- **Test/Bank Charges records removed**: {stats['test_bank_charges_removed']}
- **Final rows**: {stats['final_rows']}

## Columns Dropped
- `store_id`, `category`, `discount`, `payment_method` (100% empty)

## Data Type Corrections
- `date` cast to datetime.
- `transaction_id`, `product_id`, `region` stripped of leading/trailing whitespace.
- `customer_id` cast to integer strings (e.g. '12345'), preserving `NaN` for missing values.

## Business Decisions
- Retained postage and carriage charges (`POST`, `DOT`, `C2`).
- Retained customer returns and cancellations (represented by negative quantities and transaction IDs starting with 'C').
"""
    
    md_report_file = os.path.join(output_dir, 'data_quality_report.md')
    with open(md_report_file, 'w') as f:
        f.write(md_content)

if __name__ == '__main__':
    main()
