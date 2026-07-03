
import os
import shutil
import re

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    brain_dir = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\ef499792-a9c3-4bd3-a984-6da54fe0fe7d"
    
    demand_dir = os.path.join(root_dir, "Demand_Forecasting")
    scripts_dir = os.path.join(demand_dir, "scripts")
    datasets_dir = os.path.join(demand_dir, "datasets")
    reports_dir = os.path.join(demand_dir, "reports")
    
    # Create directories
    for d in [demand_dir, scripts_dir, datasets_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Define files to move
    scripts = [
        "create_weekly_dataset.py",
        "diagnose_forecast.py",
        "forecasting_audit.json",
        "forecasting_audit.py",
        "inspect_and_enhance_features.py",
        "inspect_daily_data.py",
        "inspect_weekly_predictions.py",
        "optimize_weekly_forecast.py",
        "remediate_leakage.py",
        "train_forecast_models.py",
        "weekly_feasibility_analysis.py"
    ]
    
    reports = [
        "actual_vs_predicted.png",
        "forecast_diagnostic_report.md",
        "forecast_evaluation_report.md",
        "forecasting_readiness_audit.md",
        "residuals_over_time.png",
        "weekly_feasibility_assessment.md",
        "weekly_forecasting_business_report.md"
    ]
    
    datasets = [
        "daily_sales_forecast_features.csv",
        "product_daily_forecast_features.csv",
        "weekly_sales_forecast_features.csv"
    ]
    
    moved_scripts = []
    moved_reports = []
    moved_datasets = []
    
    # Move scripts from scratch
    for s in scripts:
        src = os.path.join(brain_dir, "scratch", s)
        dst = os.path.join(scripts_dir, s)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            moved_scripts.append(dst)
            
    # Move reports from brain
    for r in reports:
        src = os.path.join(brain_dir, r)
        dst = os.path.join(reports_dir, r)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            moved_reports.append(dst)
            
    # Move datasets from processed_data
    for d in datasets:
        src = os.path.join(root_dir, "processed_data", d)
        dst = os.path.join(datasets_dir, d)
        if os.path.exists(src):
            shutil.move(src, dst) # The prompt says "Move (not copy)" for project files
            moved_datasets.append(dst)
            
    # Update paths in scripts
    for script_path in moved_scripts:
        if not script_path.endswith(".py"):
            continue
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Replace occurrences of processed_data\dataset.csv with Demand_Forecasting\datasets\dataset.csv
        updated_content = content.replace(
            r"processed_data\daily_sales_forecast_features.csv",
            r"Demand_Forecasting\datasets\daily_sales_forecast_features.csv"
        )
        updated_content = updated_content.replace(
            r"processed_data\weekly_sales_forecast_features.csv",
            r"Demand_Forecasting\datasets\weekly_sales_forecast_features.csv"
        )
        updated_content = updated_content.replace(
            r"processed_data/daily_sales_forecast_features.csv",
            r"Demand_Forecasting/datasets/daily_sales_forecast_features.csv"
        )
        updated_content = updated_content.replace(
            r"processed_data/weekly_sales_forecast_features.csv",
            r"Demand_Forecasting/datasets/weekly_sales_forecast_features.csv"
        )
        
        # also replace the base_path in forecasting_audit.py if needed
        updated_content = updated_content.replace(
            r"base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')",
            r"base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Demand_Forecasting/datasets')"
        )
        
        if content != updated_content:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
                
    print("Files successfully organized.")
    print("\nMoved Scripts:")
    for m in moved_scripts: print(" -", m)
    print("\nMoved Reports:")
    for m in moved_reports: print(" -", m)
    print("\nMoved Datasets:")
    for m in moved_datasets: print(" -", m)

if __name__ == "__main__":
    main()
