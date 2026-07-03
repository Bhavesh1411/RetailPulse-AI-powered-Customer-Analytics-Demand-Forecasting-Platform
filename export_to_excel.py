import pandas as pd
import os

input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
output_dir = os.path.join(input_dir, "excel_exports")
os.makedirs(output_dir, exist_ok=True)

files_to_convert = {
    'cleaned_sales_dataset.csv': 'Cleaned Sales',
    'cleaned_sales_dataset_enhanced.csv': 'Enhanced Sales',
    'customer_features.csv': 'Customer Features',
    'product_features.csv': 'Product Features',
    'daily_sales_forecast_features.csv': 'Forecast Features'
}

def format_excel_xlsxwriter(writer, sheet_name, df):
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # Freeze top row
    worksheet.freeze_panes(1, 0)
    
    # Bold headers
    header_format = workbook.add_format({'bold': True})
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)
        
    # Auto-adjust column widths (using a small sample for speed on large datasets)
    for i, col in enumerate(df.columns):
        # sample first 1000 rows to find max width quickly
        sample = df[col].head(1000).astype(str)
        col_len = sample.str.len().max()
        if pd.isna(col_len):
            col_len = 0
        header_len = len(str(col))
        max_len = max(col_len, header_len) + 2
        # Max width cap to prevent insanely wide columns
        worksheet.set_column(i, i, min(max_len, 50))

# Convert individually
for filename, sheet_name in files_to_convert.items():
    csv_path = os.path.join(input_dir, filename)
    if os.path.exists(csv_path):
        # Read the file
        print(f"Reading {filename}...")
        df = pd.read_csv(csv_path)
        
        # Create single Excel file
        excel_filename = filename.replace('.csv', '.xlsx')
        excel_path = os.path.join(output_dir, excel_filename)
        print(f"Exporting to {excel_filename}...")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            format_excel_xlsxwriter(writer, sheet_name, df)

# Create combined workbook
combined_path = os.path.join(output_dir, "RetailPulse_Data_Workbook.xlsx")
print(f"Exporting combined workbook to RetailPulse_Data_Workbook.xlsx...")
with pd.ExcelWriter(combined_path, engine='xlsxwriter') as writer:
    for filename, sheet_name in files_to_convert.items():
        csv_path = os.path.join(input_dir, filename)
        if os.path.exists(csv_path):
            print(f"Adding {sheet_name} to combined workbook...")
            df = pd.read_csv(csv_path)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            format_excel_xlsxwriter(writer, sheet_name, df)

print("Export complete.")
