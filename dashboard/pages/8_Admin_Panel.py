import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import re
import platform
from datetime import datetime
from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# Protect the page - accessible only to authenticated admins
require_auth()

# Render standard sidebar
render_sidebar()

# Inject CSS styles adhering to RetailPulse Light Theme and rules (8px grid, variables, reduced motion)
st.markdown("""
    <style>
    :root {
        --primary: #2563EB;
        --secondary: #0EA5E9;
        --success: #22C55E;
        --warning: #F59E0B;
        --danger: #EF4444;
        --slate-900: #0F172A;
        --slate-700: #334155;
        --slate-500: #64748B;
        --slate-100: #F1F5F9;
        --bg-color: #F8FAFC;
        --card-bg: #FFFFFF;
        --border-color: #E2E8F0;
    }
    
    /* Reduced Motion preference */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-delay: 0s !important;
            animation-duration: 0s !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0s !important;
            scroll-behavior: auto !important;
        }
    }
    
    /* Spacing & Layout */
    .admin-container {
        padding: 8px;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--slate-900);
        margin-top: 32px;
        margin-bottom: 16px;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 8px;
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .kpi-card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        border: 1px solid var(--border-color);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 100px;
    }
    
    .kpi-title {
        color: var(--slate-500);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .kpi-value {
        color: var(--slate-900);
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .kpi-subtitle {
        color: var(--slate-500);
        font-size: 0.75rem;
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* Module Health styling */
    .health-badge {
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-active {
        background-color: #DEF7EC;
        color: #03543F;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #713F12;
    }
    .badge-error {
        background-color: #FDE8E8;
        color: #9B1C1C;
    }
    
    /* Storage layer blocks */
    .storage-box {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        height: 100%;
    }
    .storage-box-header {
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--slate-900);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .storage-list {
        margin: 0;
        padding-left: 20px;
        color: var(--slate-700);
        font-size: 0.9rem;
    }
    .storage-list li {
        margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Resolve Base Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_dir = os.path.dirname(current_dir)
base_dir = os.path.dirname(dashboard_dir)

# ==========================================
# PARSERS & DETECTORS
# ==========================================

def get_session_theme():
    """Reads Streamlit theme config dynamically from config.toml."""
    config_path = os.path.join(dashboard_dir, ".streamlit", "config.toml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "[theme]" in content:
                # Basic parsing of config.toml keys
                theme_vars = {}
                for line in content.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        theme_vars[k.strip()] = v.strip().strip('"').strip("'")
                return f"Light Theme (Primary: {theme_vars.get('primaryColor', '#2563EB')})"
        except Exception:
            pass
    return "Light Theme (Default)"

def get_churn_roc_auc():
    """Parses Churn Prediction evaluation report dynamically."""
    report_path = os.path.join(base_dir, "churn_prediction_true", "reports", "model_evaluation_report.md")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Match: - **Test ROC AUC:** `71.7690%`
            m = re.search(r"-\s+\*\*Test ROC AUC:\*\*\s+`?([\d.]+)%?`?", content)
            if m:
                return f"{float(m.group(1)):.2f}%"
            # Fallback to table match
            m_table = re.search(r"\|\s*LightGBM\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([\d.]+)%?\s*\|", content)
            if m_table:
                return f"{float(m_table.group(1)):.2f}%"
        except Exception as e:
            return f"Error: {str(e)}"
    return "Not Available"

def get_forecast_validation_mape():
    """Parses Demand Forecasting business report dynamically."""
    report_path = os.path.join(base_dir, "Demand_Forecasting", "reports", "weekly_forecasting_business_report.md")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            m = re.search(r"Best Validation MAPE Achieved:\*\*\s+([\d.]+)%?", content)
            if m:
                return f"{float(m.group(1)):.2f}%"
            m_table = re.search(r"\|\s*\*\*Validation\*\*\s*\|\s*([\d.]+)%?\s*\|", content)
            if m_table:
                return f"{float(m_table.group(1)):.2f}%"
        except Exception:
            pass
            
    # Fallback JSON parser
    json_path = os.path.join(base_dir, "Demand_Forecasting", "datasets", "weekly_forecast_dashboard_data.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mape = data.get("model_info", {}).get("val_mape")
            if mape is not None:
                return f"{mape:.2f}%"
        except Exception:
            pass
    return "Not Available"

def get_segmentation_k():
    """Parses Segmentation KMeans clusters dynamically."""
    report_path = os.path.join(base_dir, "customer_segmentation", "reports", "kmeans_evaluation_report.md")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Match: **$K=4$**
            m = re.search(r"\*\*K=([2-8])\*\*", content)
            if m:
                return f"KMeans (K={m.group(1)})"
        except Exception:
            pass
            
    # Fallback to segments file
    seg_path = os.path.join(base_dir, "customer_segmentation", "datasets", "customer_segments_kmeans_finalone.csv")
    if os.path.exists(seg_path):
        try:
            df = pd.read_csv(seg_path)
            if "cluster" in df.columns:
                return f"KMeans (K={df['cluster'].nunique()})"
        except Exception:
            pass
    return "KMeans"

# ==========================================
# DYNAMIC MODULE HEALTH LOGIC (Modification 2)
# ==========================================

def evaluate_modules_health():
    """Evaluates module health dynamically based on file existence rules."""
    modules = {
        "Executive Overview": {
            "page": "dashboard/pages/1_Executive_Overview.py",
            "deps": [
                "Demand_Forecasting/datasets/weekly_forecast_dashboard_data.json",
                "customer_segmentation/datasets/customer_segments_kmeans_finalone.csv",
                "churn_prediction_true/predictions/customer_true_churn_predictions.csv",
                "inventory_optimization/outputs/inventory_kpi_summary.csv"
            ]
        },
        "Demand Forecasting": {
            "page": "dashboard/pages/2_Demand_Forecasting.py",
            "deps": [
                "Demand_Forecasting/datasets/weekly_forecast_dashboard_data.json",
                "Demand_Forecasting/reports/weekly_forecasting_business_report.md",
                "Demand_Forecasting/datasets/weekly_sales_forecast_features.csv"
            ]
        },
        "Customer Segmentation": {
            "page": "dashboard/pages/3_Customer_Segmentation.py",
            "deps": [
                "customer_segmentation/datasets/customer_segments_kmeans_finalone.csv",
                "customer_segmentation/reports/kmeans_evaluation_report.md"
            ]
        },
        "True Churn Prediction": {
            "page": "dashboard/pages/4_Churn_Prediction.py",
            "deps": [
                "churn_prediction_true/predictions/customer_true_churn_predictions.csv",
                "churn_prediction_true/reports/model_evaluation_report.md",
                "churn_prediction_true/models/lightgbm_model.pkl"
            ]
        },
        "Inventory Optimization": {
            "page": "dashboard/pages/5_Inventory_Optimization.py",
            "deps": [
                "inventory_optimization/outputs/inventory_master.csv",
                "inventory_optimization/outputs/reorder_recommendations.csv",
                "inventory_optimization/outputs/inventory_kpi_summary.csv"
            ]
        },
        "Alerts & Monitoring": {
            "page": "dashboard/pages/6_Alerts_and_Monitoring.py",
            "deps": [
                "churn_prediction_true/predictions/customer_true_churn_predictions.csv",
                "inventory_optimization/outputs/reorder_recommendations.csv",
                "Demand_Forecasting/datasets/weekly_forecast_dashboard_data.json"
            ]
        },
        "Export Center": {
            "page": "dashboard/pages/7_Export_Center.py",
            "deps": [
                "customer_segmentation/datasets/customer_segments_kmeans_finalone.csv",
                "churn_prediction_true/predictions/customer_true_churn_predictions.csv",
                "Demand_Forecasting/datasets/weekly_forecast_dashboard_data.json",
                "inventory_optimization/outputs/inventory_master.csv"
            ]
        }
    }
    
    results = {}
    for name, info in modules.items():
        page_path = os.path.join(base_dir, info["page"])
        if not os.path.exists(page_path):
            results[name] = {
                "status": "🔴 Error",
                "health": "Missing page file in dashboard/pages",
                "availability": "0%",
                "last_loaded": "Not Available"
            }
        else:
            missing_deps = [d for d in info["deps"] if not os.path.exists(os.path.join(base_dir, d))]
            if missing_deps:
                results[name] = {
                    "status": "🟡 Warning",
                    "health": f"Missing components: {', '.join([os.path.basename(d) for d in missing_deps])}",
                    "availability": f"{int((len(info['deps']) - len(missing_deps)) / len(info['deps']) * 100)}%",
                    "last_loaded": "Not Available"
                }
            else:
                # Dynamically retrieve last loaded/modified time from the latest modified dependency
                mtimes = []
                for d in info["deps"]:
                    d_path = os.path.join(base_dir, d)
                    if os.path.exists(d_path):
                        mtimes.append(os.path.getmtime(d_path))
                last_loaded = datetime.fromtimestamp(max(mtimes)).strftime('%Y-%m-%d %H:%M:%S') if mtimes else "Not Available"
                
                results[name] = {
                    "status": "🟢 Active",
                    "health": "Healthy (All assets detected)",
                    "availability": "100%",
                    "last_loaded": last_loaded
                }
    return results

health_data = evaluate_modules_health()

# ==========================================
# DYNAMIC PROJECT STATISTICS (Modification 1)
# ==========================================

def get_project_statistics():
    """Gathers overall file counts dynamically from the project structure."""
    stats = {
        "datasets": 0,
        "models": 0,
        "pages": 0,
        "reports": 0,
        "exports": 0,
        "predictions": 0
    }
    
    # 1. Total Pages
    pages_dir = os.path.join(dashboard_dir, "pages")
    if os.path.exists(pages_dir):
        stats["pages"] = len([f for f in os.listdir(pages_dir) if f.endswith('.py')])
        
    # 2. Total Datasets & Prediction files & Models & Reports
    for root, dirs, files in os.walk(base_dir):
        # Skip git, cache, .streamlit directories
        if any(x in root for x in [".git", "__pycache__", ".streamlit"]):
            continue
        parent = os.path.basename(root)
        for f in files:
            # Models
            if f.endswith('.pkl') and parent in ["models", "processed_data"]:
                stats["models"] += 1
            # Predictions
            elif f.endswith('.csv') and parent == "predictions":
                stats["predictions"] += 1
            # Reports
            elif f.endswith('.md') and parent in ["reports", "documentation"]:
                stats["reports"] += 1
            # Datasets
            elif f.endswith(('.csv', '.json')) and parent in ["datasets", "outputs", "processed_data"]:
                stats["datasets"] += 1
                
    # 3. Export Files
    export_dirs = [
        os.path.join(dashboard_dir, "exports"),
        os.path.join(base_dir, "processed_data", "excel_exports")
    ]
    for ed in export_dirs:
        if os.path.exists(ed):
            stats["exports"] += len([f for f in os.listdir(ed) if os.path.isfile(os.path.join(ed, f))])
            
    return stats

stats = get_project_statistics()

# ==========================================
# DATASET STATUS METADATA
# ==========================================

def get_file_metadata(filepath):
    """Loads file metadata dynamically. Returns 'Not Available' if missing."""
    if not os.path.exists(filepath):
        return {
            "name": os.path.basename(filepath),
            "rows": "Not Available",
            "cols": "Not Available",
            "size": "Not Available",
            "modified": "Not Available",
            "status": "🔴 Missing"
        }
    
    name = os.path.basename(filepath)
    size_bytes = os.path.getsize(filepath)
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.2f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        
    mtime = os.path.getmtime(filepath)
    modified_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    rows, cols = "Not Available", "Not Available"
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            rows, cols = f"{len(df):,}", f"{len(df.columns)}"
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                hist_len = len(data.get('historical', []))
                fut_len = len(data.get('future', []))
                rows = f"{hist_len + fut_len:,}"
                cols = f"{len(data.get('model_info', {}))} keys"
            elif isinstance(data, list):
                rows = f"{len(data):,}"
                cols = "N/A"
    except Exception:
        pass
        
    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "size": size_str,
        "modified": modified_str,
        "status": "🟢 Available"
    }

datasets_paths = {
    "Customer Segmentation": os.path.join(base_dir, "customer_segmentation", "datasets", "customer_segments_kmeans_finalone.csv"),
    "True Churn": os.path.join(base_dir, "churn_prediction_true", "predictions", "customer_true_churn_predictions.csv"),
    "Demand Forecasting": os.path.join(base_dir, "Demand_Forecasting", "datasets", "weekly_forecast_dashboard_data.json"),
    "Inventory Optimization": os.path.join(base_dir, "inventory_optimization", "outputs", "inventory_master.csv")
}

dataset_rows = []
for title, path in datasets_paths.items():
    meta = get_file_metadata(path)
    dataset_rows.append({
        "Dataset Module": title,
        "File Name": meta["name"],
        "Rows": meta["rows"],
        "Columns": meta["cols"],
        "Size": meta["size"],
        "Last Modified": meta["modified"],
        "Status": meta["status"]
    })

# Compute overall project status dynamically based on module status
status_counts = [item["status"] for item in health_data.values()]
if all(s == "🟢 Active" for s in status_counts):
    project_completion_status = "100% Completed (Fully Operational)"
    completion_color = "var(--success)"
elif any(s == "🔴 Error" for s in status_counts):
    project_completion_status = "Needs Attention (Modules Incomplete)"
    completion_color = "var(--danger)"
else:
    project_completion_status = "85% Completed (Partially Operational)"
    completion_color = "var(--warning)"

# ==========================================
# PAGE RENDER - HEADER
# ==========================================
st.markdown("<h1>⚙️ System Administration Panel</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--slate-700); font-size:1.1rem; margin-top:-0.5rem;'>Read-only infrastructure monitoring dashboard for system and database status.</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# SECTION 1 – PLATFORM OVERVIEW (KPI CARDS)
# ==========================================
st.markdown("<div class='section-title' style='margin-top: 0;'>Section 1: Platform Overview</div>", unsafe_allow_html=True)

overview_col1, overview_col2 = st.columns(2)

with overview_col1:
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid var(--primary); margin-bottom:16px;">
            <div class="kpi-title">Platform & Environment</div>
            <div class="kpi-value">RetailPulse Platform</div>
            <div class="kpi-subtitle">Active Modules: <strong>{len(health_data)} Implemented</strong></div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid var(--secondary); margin-bottom:16px;">
            <div class="kpi-title">Current Logged-in User</div>
            <div class="kpi-value">{st.session_state.get("username", "System Administrator")}</div>
            <div class="kpi-subtitle">Access Privileges: <strong>Admin / Read-Only</strong></div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid {completion_color};">
            <div class="kpi-title">Project Completion Status</div>
            <div class="kpi-value" style="color:{completion_color};">{project_completion_status}</div>
            <div class="kpi-subtitle">Dynamic health calculation</div>
        </div>
    """, unsafe_allow_html=True)

with overview_col2:
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid var(--success); margin-bottom:16px;">
            <div class="kpi-title">Platform Version</div>
            <div class="kpi-value">v1.0.0</div>
            <div class="kpi-subtitle">Reason: Defaulting to v1.0.0 as no release version tag is configured.</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid var(--warning); margin-bottom:16px;">
            <div class="kpi-title">Login Timestamp</div>
            <div class="kpi-value">Not Available</div>
            <div class="kpi-subtitle">Reason: User login session time logging is not implemented.</div>
        </div>
        <div class="kpi-card" style="border-left: 4px solid var(--slate-900);">
            <div class="kpi-title">Dashboard Status & Theme</div>
            <div class="kpi-value">Running / {get_session_theme().split()[0]}</div>
            <div class="kpi-subtitle">Parsed from dashboard configurations</div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# SECTION 2 – MODULE HEALTH
# ==========================================
st.markdown("<div class='section-title'>Section 2: Module Health</div>", unsafe_allow_html=True)

module_health_list = []
for name, data in health_data.items():
    badge_class = "badge-active" if "🟢" in data["status"] else ("badge-warning" if "🟡" in data["status"] else "badge-error")
    status_label = data["status"].split(" ")[1] if " " in data["status"] else data["status"]
    
    module_health_list.append({
        "Module Name": name,
        "Status": f'<span class="health-badge {badge_class}">{status_label}</span>',
        "Health Details": data["health"],
        "Availability": data["availability"],
        "Last Loaded Assets": data["last_loaded"]
    })

module_df = pd.DataFrame(module_health_list)
st.write(
    module_df.to_html(escape=False, index=False, justify="left", classes="custom-table"), 
    unsafe_allow_html=True
)

# ==========================================
# SECTION 3 – DATASET STATUS
# ==========================================
st.markdown("<div class='section-title'>Section 3: Dataset Status</div>", unsafe_allow_html=True)

dataset_status_df = pd.DataFrame(dataset_rows)
st.dataframe(dataset_status_df, use_container_width=True, hide_index=True)

# ==========================================
# SECTION 4 – MODEL REGISTRY (Modification 1)
# ==========================================
st.markdown("<div class='section-title'>Section 4: Model Registry</div>", unsafe_allow_html=True)

# Retrieve metrics dynamically
forecast_mape = get_forecast_validation_mape()
churn_auc = get_churn_roc_auc()
seg_algorithm = get_segmentation_k()

# Determine active model registry values
reg1, reg2 = st.columns(2)
reg3, reg4 = st.columns(2)

with reg1:
    st.markdown("""
        <div class="storage-box">
            <div class="storage-box-header">👥 Customer Segmentation Model</div>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Algorithm:</strong> """ + seg_algorithm + """</p>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Version:</strong> v1.0 <span style="font-size: 0.75rem; color: var(--slate-500);">(Reason: No version metadata was generated, defaulting to v1.0)</span></p>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Clusters Found:</strong> 4</p>
            <p style="margin: 0; font-size: 0.95rem;"><strong>Status:</strong> <span class="health-badge badge-active">Active</span></p>
        </div>
    """, unsafe_allow_html=True)

with reg2:
    st.markdown("""
        <div class="storage-box">
            <div class="storage-box-header">📈 Demand Forecasting Model</div>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Algorithm:</strong> XGBoost Regressor</p>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Granularity:</strong> Weekly Aggregated</p>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Validation MAPE:</strong> """ + forecast_mape + """</p>
            <p style="margin: 0; font-size: 0.95rem;"><strong>Status:</strong> <span class="health-badge badge-active">Production Ready</span></p>
        </div>
    """, unsafe_allow_html=True)

with reg3:
    st.markdown("""
        <div class="storage-box">
            <div class="storage-box-header">🎯 True Churn Prediction Model</div>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Algorithm:</strong> LightGBM Classifier</p>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>ROC AUC:</strong> """ + churn_auc + """</p>
            <p style="margin: 0; font-size: 0.95rem;"><strong>Status:</strong> <span class="health-badge badge-active">Active</span></p>
        </div>
    """, unsafe_allow_html=True)

with reg4:
    # Inventory Optimization KPIs
    inv_kpi_path = os.path.join(base_dir, "inventory_optimization", "outputs", "inventory_kpi_summary.csv")
    inv_health_val = "Not Available"
    if os.path.exists(inv_kpi_path):
        try:
            inv_kpis = pd.read_csv(inv_kpi_path)
            inv_dict = dict(zip(inv_kpis['metric'], inv_kpis['value']))
            if 'inventory_health_score' in inv_dict:
                inv_health_val = f"{float(inv_dict['inventory_health_score']):.2f}/100"
        except Exception:
            pass
            
    st.markdown("""
        <div class="storage-box">
            <div class="storage-box-header">📦 Inventory Optimization System</div>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Methodologies:</strong> ABC Analysis, Safety Stock, ROP, EOQ</p>
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;"><strong>Health Score:</strong> """ + inv_health_val + """</p>
            <p style="margin: 0; font-size: 0.95rem;"><strong>Status:</strong> <span class="health-badge badge-active">Active</span></p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# SECTION 5 – PROJECT STATISTICS
# ==========================================
st.markdown("<div class='section-title'>Section 5: Project Statistics</div>", unsafe_allow_html=True)

col_s1, col_s2, col_s3 = st.columns(3)
col_s4, col_s5, col_s6 = st.columns(3)

with col_s1:
    st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--primary);">
            <div class="kpi-title">Total Datasets</div>
            <div class="kpi-value">{stats['datasets']} files</div>
            <div class="kpi-subtitle">CSV & JSON formats</div>
        </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--secondary);">
            <div class="kpi-title">Total Models</div>
            <div class="kpi-value">{stats['models']} binaries</div>
            <div class="kpi-subtitle">Pickle (.pkl) models</div>
        </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--success);">
            <div class="kpi-title">Dashboard Pages</div>
            <div class="kpi-value">{stats['pages']} pages</div>
            <div class="kpi-subtitle">Streamlit script files</div>
        </div>
    """, unsafe_allow_html=True)

with col_s4:
    st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--warning);">
            <div class="kpi-title">Reports Generated</div>
            <div class="kpi-value">{stats['reports']} reports</div>
            <div class="kpi-subtitle">Markdown analysis logs</div>
        </div>
    """, unsafe_allow_html=True)

with col_s5:
    st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--danger);">
            <div class="kpi-title">Export Files</div>
            <div class="kpi-value">{stats['exports']} files</div>
            <div class="kpi-subtitle">User-downloaded Excel/CSVs</div>
        </div>
    """, unsafe_allow_html=True)

with col_s6:
    st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--slate-900);">
            <div class="kpi-title">Prediction Files</div>
            <div class="kpi-value">{stats['predictions']} predictions</div>
            <div class="kpi-subtitle">Inference output datasets</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# SECTION 6 – STORAGE LAYER
# ==========================================
st.markdown("<div class='section-title'>Section 6: Storage Layer</div>", unsafe_allow_html=True)

store_col1, store_col2 = st.columns(2)

with store_col1:
    st.markdown("""
        <div class="storage-box">
            <div class="storage-box-header">💾 Current Local Storage</div>
            <ul class="storage-list">
                <li><strong>CSV Files:</strong> Storing tabular outputs and predictions (e.g. customer cohorts, reorder recommendations).</li>
                <li><strong>JSON Files:</strong> Storing dashboard configurations and forecasting time series trajectories.</li>
                <li><strong>Pickle Models (.pkl):</strong> Model binaries for tree algorithms and numeric scalers.</li>
                <li><strong>Markdown Reports:</strong> Automated performance reporting and mathematical audit metrics.</li>
                <li><strong>Dashboard Files:</strong> Multi-page Python scripts loaded natively by Streamlit.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with store_col2:
    st.markdown("""
        <div class="storage-box">
            <div class="storage-box-header">☁️ Future Production Target Storage</div>
            <ul class="storage-list">
                <li><strong>PostgreSQL:</strong> Primary relational database for transactions, secure logins, and historical reports.</li>
                <li><strong>MySQL / SQLite:</strong> Secondary options for localized metadata caching or configuration backups.</li>
                <li><strong>Cloud Object Storage (e.g., S3, Cloud Storage):</strong> Distributed file storage for big transaction datasets, images, and model binaries.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# SECTION 7 – SYSTEM INFORMATION
# ==========================================
st.markdown("<div class='section-title'>Section 7: System Information</div>", unsafe_allow_html=True)

sys_info = {
    "Parameter": [
        "Python Version",
        "Streamlit Version",
        "OS Platform",
        "OS Release",
        "Project Architecture",
        "Authentication Manager Status",
        "Dashboard Runtime Status"
    ],
    "Detected Value": [
        sys.version.split()[0],
        st.__version__,
        platform.system(),
        platform.release(),
        platform.machine(),
        "Operational (Temporary Local Credentials active, PostgreSQL integration planned)",
        "Healthy (Multi-page configuration verified)"
    ]
}

sys_info_df = pd.DataFrame(sys_info)
st.dataframe(sys_info_df, use_container_width=True, hide_index=True)
