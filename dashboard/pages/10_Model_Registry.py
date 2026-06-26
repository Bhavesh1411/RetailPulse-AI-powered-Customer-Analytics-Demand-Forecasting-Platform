import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# Protect the page
require_auth()

# Render standard sidebar
render_sidebar()

# Inject Custom CSS following RetailPulse Light Theme rules (8px grid, CSS variables, reduced motion)
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
    .registry-container {
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
    
    /* Metric Grid */
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
    
    /* Artifact list panel styles */
    .artifact-box {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .artifact-header {
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--slate-900);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .artifact-list {
        margin: 0;
        padding: 0;
        list-style: none;
    }
    
    .artifact-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid var(--slate-100);
        font-size: 0.9rem;
    }
    
    .artifact-item:last-child {
        border-bottom: none;
    }
    
    /* Health Badge styles */
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
    
    /* Comparison Table styles */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 24px;
    }
    .custom-table th, .custom-table td {
        padding: 12px 16px;
        border-bottom: 1px solid var(--border-color);
        text-align: left;
    }
    .custom-table th {
        background-color: var(--slate-100);
        color: var(--slate-900);
        font-weight: 600;
    }
    .custom-table tr:hover {
        background-color: var(--slate-100);
    }
    </style>
""", unsafe_allow_html=True)

# Resolve Base Paths
current_dir = os.path.dirname(os.path.abspath(__file__)) 
dashboard_dir = os.path.dirname(current_dir)
base_dir = os.path.dirname(dashboard_dir)

# ==========================================
# PARSERS & ARTIFACT CHECKERS (Modification 1 & 2)
# ==========================================

def check_file_status(relative_path):
    """Physically checks for file existence and returns status metrics."""
    full_path = os.path.join(base_dir, relative_path)
    exists = os.path.exists(full_path)
    badge = "🟢 Exists" if exists else "🔴 Missing"
    return exists, badge

# Define artifact maps
artifact_registry = {
    "segmentation": {
        "dataset": "customer_segmentation/datasets/customer_segments_kmeans_finalone.csv",
        "report": "customer_segmentation/reports/kmeans_evaluation_report.md"
    },
    "churn": {
        "model": "churn_prediction_true/models/lightgbm_model.pkl",
        "report": "churn_prediction_true/reports/model_evaluation_report.md",
        "predictions": "churn_prediction_true/predictions/customer_true_churn_predictions.csv"
    },
    "forecasting": {
        "json": "Demand_Forecasting/datasets/weekly_forecast_dashboard_data.json",
        "report": "Demand_Forecasting/reports/weekly_forecasting_business_report.md"
    },
    "inventory": {
        "master": "inventory_optimization/outputs/inventory_master.csv",
        "kpi": "inventory_optimization/outputs/inventory_kpi_summary.csv"
    }
}

# Evaluate all artifact presence states
health_status = {}
for module, files in artifact_registry.items():
    health_status[module] = {}
    for key, rel_path in files.items():
        exists, badge = check_file_status(rel_path)
        health_status[module][key] = {
            "path": rel_path,
            "exists": exists,
            "badge": badge
        }

# Dynamic loaders with try-except wraps (Modification 1)
def load_churn_metrics():
    metrics = {
        "accuracy": "Not Available",
        "precision": "Not Available",
        "recall": "Not Available",
        "f1_score": "Not Available",
        "roc_auc": "Not Available"
    }
    report_path = os.path.join(base_dir, artifact_registry["churn"]["report"])
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Split to look in the Test Set table specifically
            parts = content.split("## 2. Test Set Performance")
            if len(parts) > 1:
                test_section = parts[1]
                m_lgb = re.search(
                    r"\|\s*LightGBM\s*\|\s*([\d.]+)%?\s*\|\s*([\d.]+)%?\s*\|\s*([\d.]+)%?\s*\|\s*([\d.]+)%?\s*\|\s*([\d.]+)%?\s*\|",
                    test_section
                )
                if m_lgb:
                    metrics["accuracy"] = f"{float(m_lgb.group(1)):.2f}%"
                    metrics["precision"] = f"{float(m_lgb.group(2)):.2f}%"
                    metrics["recall"] = f"{float(m_lgb.group(3)):.2f}%"
                    metrics["f1_score"] = f"{float(m_lgb.group(4)):.2f}%"
                    metrics["roc_auc"] = f"{float(m_lgb.group(5)):.2f}%"
            # General fallback regex if table split search yielded nothing
            if metrics["roc_auc"] == "Not Available":
                m_auc = re.search(r"-\s+\*\*Test ROC AUC:\*\*\s+`?([\d.]+)%?`?", content)
                if m_auc: metrics["roc_auc"] = f"{float(m_auc.group(1)):.2f}%"
                m_recall = re.search(r"-\s+\*\*Test Recall:\*\*\s+`?([\d.]+)%?`?", content)
                if m_recall: metrics["recall"] = f"{float(m_recall.group(1)):.2f}%"
                m_f1 = re.search(r"-\s+\*\*Test F1-Score:\*\*\s+`?([\d.]+)%?`?", content)
                if m_f1: metrics["f1_score"] = f"{float(m_f1.group(1)):.2f}%"
                m_acc = re.search(r"-\s+\*\*Test Accuracy:\*\*\s+`?([\d.]+)%?`?", content)
                if m_acc: metrics["accuracy"] = f"{float(m_acc.group(1)):.2f}%"
        except Exception:
            pass
    return metrics

def load_forecast_metrics():
    metrics = {
        "val_mape": "Not Available",
        "test_mape": "Not Available"
    }
    json_path = os.path.join(base_dir, artifact_registry["forecasting"]["json"])
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            info = data.get("model_info", {})
            val_mape = info.get("val_mape")
            test_mape = info.get("test_mape")
            if val_mape is not None:
                metrics["val_mape"] = f"{float(val_mape):.2f}%"
            if test_mape is not None:
                metrics["test_mape"] = f"{float(test_mape):.2f}%"
        except Exception:
            pass
            
    # Fallback to business report parser
    if metrics["val_mape"] == "Not Available":
        report_path = os.path.join(base_dir, artifact_registry["forecasting"]["report"])
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    content = f.read()
                m_val = re.search(r"Best Validation MAPE Achieved:\*\*\s+([\d.]+)%?", content)
                if m_val:
                    metrics["val_mape"] = f"{float(m_val.group(1)):.2f}%"
            except Exception:
                pass
    return metrics

def load_segmentation_metrics():
    metrics = {
        "silhouette": "Not Available",
        "clusters_count": "Not Available",
        "dataset_size": "Not Available",
        "training_date": "Not Available"
    }
    report_path = os.path.join(base_dir, artifact_registry["segmentation"]["report"])
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Silhouette Score without active_months is 0.3250
            m_sil = re.search(r"\|\s*\*\*4\*\*\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|", content)
            if m_sil:
                metrics["silhouette"] = f"{float(m_sil.group(1)):.4f}"
            m_k = re.search(r"\*\*K=([2-8])\*\*", content)
            if m_k:
                metrics["clusters_count"] = m_k.group(1)
        except Exception:
            pass
            
    # Extract size and true count from dataset
    seg_csv = os.path.join(base_dir, artifact_registry["segmentation"]["dataset"])
    if os.path.exists(seg_csv):
        try:
            df = pd.read_csv(seg_csv)
            metrics["dataset_size"] = f"{len(df):,} customers"
            if "cluster" in df.columns:
                metrics["clusters_count"] = str(df["cluster"].nunique())
            mtime = os.path.getmtime(seg_csv)
            metrics["training_date"] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
        except Exception:
            pass
    return metrics

def load_inventory_metrics():
    metrics = {
        "health_score": "Not Available",
        "dataset_size": "Not Available"
    }
    kpi_path = os.path.join(base_dir, artifact_registry["inventory"]["kpi"])
    if os.path.exists(kpi_path):
        try:
            kpis = pd.read_csv(kpi_path)
            inv_dict = dict(zip(kpis['metric'], kpis['value']))
            if 'inventory_health_score' in inv_dict:
                metrics["health_score"] = f"{float(inv_dict['inventory_health_score']):.2f}/100"
            if 'total_products' in inv_dict:
                metrics["dataset_size"] = f"{int(float(inv_dict['total_products'])):,} products"
        except Exception:
            pass
    return metrics

# Load all parsed metric dicts
churn_m = load_churn_metrics()
forecast_m = load_forecast_metrics()
segmentation_m = load_segmentation_metrics()
inventory_m = load_inventory_metrics()

# ==========================================
# PAGE HEADER
# ==========================================
st.markdown("<h1>🛡️ Model Registry & Health Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--slate-700); font-size:1.1rem; margin-top:-0.5rem;'>Unified model management control room verifying physical artifact states, performance metrics, and operational readiness.</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# SECTION 1 – PLATFORM MODELS OVERVIEW
# ==========================================
st.markdown("<div class='section-title' style='margin-top: 0;'>Section 1: Platform Models Overview</div>", unsafe_allow_html=True)

# Calculate active / healthy counts based on artifact checks
active_models_count = 0
total_models = 4

# Checks for segmentation
if health_status["segmentation"]["dataset"]["exists"] and health_status["segmentation"]["report"]["exists"]:
    active_models_count += 1
# Checks for churn
if health_status["churn"]["model"]["exists"] and health_status["churn"]["report"]["exists"] and health_status["churn"]["predictions"]["exists"]:
    active_models_count += 1
# Checks for forecast
if health_status["forecasting"]["json"]["exists"] and health_status["forecasting"]["report"]["exists"]:
    active_models_count += 1
# Checks for inventory
if health_status["inventory"]["master"]["exists"] and health_status["inventory"]["kpi"]["exists"]:
    active_models_count += 1

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

with overview_col1:
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid var(--primary);">
            <div class="kpi-title">Total Registered Models</div>
            <div class="kpi-value">{total_models}</div>
            <div class="kpi-subtitle">Analytical Engines</div>
        </div>
    """, unsafe_allow_html=True)

with overview_col2:
    health_color = "var(--success)" if active_models_count == total_models else ("var(--warning)" if active_models_count > 0 else "var(--danger)")
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {health_color};">
            <div class="kpi-title">Active & Healthy Models</div>
            <div class="kpi-value" style="color: {health_color};">{active_models_count} / {total_models}</div>
            <div class="kpi-subtitle">Verified Artifact Health</div>
        </div>
    """, unsafe_allow_html=True)

with overview_col3:
    # All are deployed in dashboard
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid var(--secondary);">
            <div class="kpi-title">Production Ready</div>
            <div class="kpi-value">{active_models_count}</div>
            <div class="kpi-subtitle">Integrated to Dashboard</div>
        </div>
    """, unsafe_allow_html=True)

with overview_col4:
    # Let's count dashboard modules using models (Demand, Segmentation, Churn, Inventory, Alerts, Prediction Center)
    modules_count = 6
    st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid var(--slate-900);">
            <div class="kpi-title">Integrated Modules</div>
            <div class="kpi-value">{modules_count} Modules</div>
            <div class="kpi-subtitle">Serving Active Pages</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# SECTIONS 2 - 5: INDIVIDUAL MODEL BLOCKS
# ==========================================
model_tabs = st.tabs([
    "👥 Customer Segmentation",
    "🎯 True Churn Prediction",
    "📈 Demand Forecasting",
    "📦 Inventory Heuristics"
])

# Helper for artifact badge rendering
def render_artifact_checks(checks):
    st.markdown("##### 📁 Artifact Verification Registry")
    for key, item in checks.items():
        badge_class = "badge-active" if item["exists"] else "badge-error"
        badge_label = "Exists" if item["exists"] else "Missing"
        icon = "🟢" if item["exists"] else "🔴"
        
        st.markdown(f"""
            <div class="artifact-item">
                <span><code>{item['path']}</code></span>
                <span class="health-badge {badge_class}">{icon} {badge_label}</span>
            </div>
        """, unsafe_allow_html=True)

# -----------------
# TAB 2: Customer Segmentation Model
# -----------------
with model_tabs[0]:
    st.markdown("### Section 2: Customer Segmentation Model")
    
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("##### ⚙️ Model Specifications")
        spec_df = pd.DataFrame([
            {"Parameter": "Model Name", "Value": "Customer Segmentation Model"},
            {"Parameter": "Algorithm", "Value": "KMeans Clustering"},
            {"Parameter": "Purpose", "Value": "Groups customers with similar purchasing behavior."},
            {"Parameter": "Dataset Used", "Value": "processed_data/customer_features.csv"},
            {"Parameter": "Standardized Features Used", "Value": "6 Variables"},
            {"Parameter": "Number of Clusters (K)", "Value": segmentation_m["clusters_count"]},
            {"Parameter": "Silhouette Score", "Value": segmentation_m["silhouette"]},
            {"Parameter": "Output Dataset File", "Value": "customer_segmentation/datasets/customer_segments_kmeans_finalone.csv"},
            {"Parameter": "Dataset Size", "Value": segmentation_m["dataset_size"]},
            {"Parameter": "Training/Modified Date", "Value": segmentation_m["training_date"]},
            {"Parameter": "Version", "Value": "v1.0"},
            {"Parameter": "Status", "Value": "Active / Deployed"}
        ])
        st.dataframe(spec_df, use_container_width=True, hide_index=True)
        
    with col_r:
        st.markdown('<div class="artifact-box">', unsafe_allow_html=True)
        render_artifact_checks(health_status["segmentation"])
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------
# TAB 3: True Churn Prediction Model
# -----------------
with model_tabs[1]:
    st.markdown("### Section 3: True Churn Prediction Model")
    
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("##### ⚙️ Model Specifications")
        spec_df = pd.DataFrame([
            {"Parameter": "Model Name", "Value": "True Churn Prediction Model"},
            {"Parameter": "Algorithm", "Value": "LightGBM Classifier"},
            {"Parameter": "Purpose", "Value": "Predicts customers likely to stop purchasing."},
            {"Parameter": "Dataset Used", "Value": "processed_data/churn_model_dataset.csv"},
            {"Parameter": "Accuracy", "Value": churn_m["accuracy"]},
            {"Parameter": "Precision", "Value": churn_m["precision"]},
            {"Parameter": "Recall", "Value": churn_m["recall"]},
            {"Parameter": "F1 Score", "Value": churn_m["f1_score"]},
            {"Parameter": "ROC AUC", "Value": churn_m["roc_auc"]},
            {"Parameter": "Features Used", "Value": "Recency, Frequency, Monetary, AOV, Tenure"},
            {"Parameter": "Prediction Output File", "Value": "churn_prediction_true/predictions/customer_true_churn_predictions.csv"},
            {"Parameter": "Version", "Value": "v1.0"},
            {"Parameter": "Status", "Value": "Active / Deployed"}
        ])
        st.dataframe(spec_df, use_container_width=True, hide_index=True)
        
    with col_r:
        st.markdown('<div class="artifact-box">', unsafe_allow_html=True)
        render_artifact_checks(health_status["churn"])
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------
# TAB 4: Demand Forecasting Model
# -----------------
with model_tabs[2]:
    st.markdown("### Section 4: Demand Forecasting Model")
    
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("##### ⚙️ Model Specifications")
        spec_df = pd.DataFrame([
            {"Parameter": "Model Name", "Value": "Demand Forecasting Model"},
            {"Parameter": "Algorithm", "Value": "XGBoost Regressor + Holiday Features"},
            {"Parameter": "Forecast Type", "Value": "Weekly Aggregated Sales Forecast"},
            {"Parameter": "Forecast Horizon", "Value": "8 Weeks"},
            {"Parameter": "Granularity", "Value": "Weekly"},
            {"Parameter": "Validation MAPE", "Value": forecast_m["val_mape"]},
            {"Parameter": "Test MAPE", "Value": forecast_m["test_mape"]},
            {"Parameter": "Dataset Used", "Value": "processed_data/cleaned_sales_dataset.csv"},
            {"Parameter": "Features Used", "Value": "Lag Features, Rolling Mean, Holiday Features, Trend Features"},
            {"Parameter": "Forecast Output File", "Value": "Demand_Forecasting/datasets/weekly_forecast_dashboard_data.json"},
            {"Parameter": "Status", "Value": "Production Ready / Active"}
        ])
        st.dataframe(spec_df, use_container_width=True, hide_index=True)
        
    with col_r:
        st.markdown('<div class="artifact-box">', unsafe_allow_html=True)
        render_artifact_checks(health_status["forecasting"])
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------
# TAB 5: Inventory Optimization Engine
# -----------------
with model_tabs[3]:
    st.markdown("### Section 5: Inventory Optimization Engine")
    
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("##### ⚙️ Engine Specifications")
        st.markdown("""
            Unlike machine learning classification or regression pipelines, the Inventory Optimization Engine 
            utilizes deterministic operations research heuristics to minimize replenishment risks and storage costs.
        """)
        spec_df = pd.DataFrame([
            {"Heuristic Component", "Mathematical / Operations Research Methodology"},
            {"ABC Analysis", "Revenue-based sorting classifying products into Class A (80% rev), B (15% rev), and C (5% rev)."},
            {"Safety Stock", "Calculated as: Z-score * Demand Std Dev * sqrt(Lead Time) to buffer lead time fluctuations."},
            {"Reorder Point (ROP)", "Calculated as: (Average Daily Demand * Lead Time) + Safety Stock to trigger purchase cycles."},
            {"Economic Order Qty (EOQ)", "Calculated as: sqrt((2 * Annual Demand * Order Cost) / Holding Cost) to optimize orders."},
            {"Stockout Risk", "Probability computed from demand probability density integrations over the lead time period."},
            {"Inventory Health Score", f"Aggregated benchmark score. Current: {inventory_m['health_score']}"},
            {"Input Dataset Used", "processed_data/cleaned_sales_dataset.csv"},
            {"Output Master File", "inventory_optimization/outputs/inventory_master.csv"},
            {"Output KPI Summary File", "inventory_optimization/outputs/inventory_kpi_summary.csv"},
            {"Status", "Active / Deployed"}
        ])
        # Force column names and render
        spec_df.columns = ["Heuristic Module", "Methodology / Operational Details"]
        st.dataframe(spec_df, use_container_width=True, hide_index=True)
        
    with col_r:
        st.markdown('<div class="artifact-box">', unsafe_allow_html=True)
        render_artifact_checks(health_status["inventory"])
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# SECTION 6 – FEATURE REGISTRY
# ==========================================
st.markdown("<div class='section-title'>Section 6: Feature Registry</div>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--slate-500); font-size:0.9rem; margin-top:-0.5rem; margin-bottom:16px;'>Important model inputs and feature descriptions categorized by analytical modules.</p>", unsafe_allow_html=True)

f_col1, f_col2 = st.columns(2)
with f_col1:
    with st.expander("👥 Customer Segmentation Features"):
        st.markdown("""
            * **recency**: Days elapsed since the customer's last order placement.
            * **frequency**: Total count of unique orders placed over customer lifetime.
            * **monetary**: Total value ($) spent across all purchase cycles.
            * **average_purchase_value**: The average order basket value ($) computed per transaction.
            * **customer_tenure**: Active lifespan of the customer account in days.
            * **weekend_sales_ratio**: The proportion of purchases completed on Saturdays/Sundays versus weekdays.
        """)
        
    with st.expander("🎯 True Churn Prediction Features"):
        st.markdown("""
            * **recency**: Days elapsed since the customer's last order placement.
            * **frequency**: Total count of unique orders placed over customer lifetime.
            * **monetary**: Total value ($) spent across all purchase cycles.
            * **average_purchase_value (AOV)**: Mean revenue generated per order transaction.
            * **customer_tenure**: Customer tenure in days since initial record creation.
            * **customer_loyalty_score**: Multi-factor customer engagement score.
        """)

with f_col2:
    with st.expander("📈 Demand Forecasting Features"):
        st.markdown("""
            * **lag_features**: Historical revenue values (e.g., lag_1, lag_2 weeks) feeding auto-regressive patterns.
            * **rolling_mean**: Moving averages (4-week and 8-week windows) smoothing high-frequency noise.
            * **holiday_features**: Binary flag indicators flagging critical holiday sales spikes (Q4 spikes).
            * **trend_features**: Numeric timeline steps capturing long-term market growth or contraction.
        """)
        
    with st.expander("📦 Inventory Optimization Inputs"):
        st.markdown("""
            * **avg_daily_demand**: Daily mean quantity sold over the observation timeline.
            * **demand_std_dev**: Daily demand variance capturing sales volatility.
            * **eoq_parameters**: Composite factors consisting of transactional ordering fees ($), unit prices, and unit holding rates.
            * **safety_stock**: Safety buffers incorporating lead times, service level coefficients, and demand volatility.
        """)

# ==========================================
# SECTION 7 – MODEL PERFORMANCE COMPARISON
# ==========================================
st.markdown("<div class='section-title'>Section 7: Model Performance Comparison</div>", unsafe_allow_html=True)

# Build metrics table
comparison_rows = [
    {
        "Module": "Customer Segmentation",
        "Algorithm": "KMeans Clustering",
        "Main Metric": "Silhouette Score",
        "Current Value": segmentation_m["silhouette"],
        "Status": "🟢 Active"
    },
    {
        "Module": "Demand Forecasting",
        "Algorithm": "XGBoost Regressor",
        "Main Metric": "Validation MAPE",
        "Current Value": forecast_m["val_mape"],
        "Status": "🟢 Production Ready"
    },
    {
        "Module": "True Churn Prediction",
        "Algorithm": "LightGBM Classifier",
        "Main Metric": "ROC AUC",
        "Current Value": churn_m["roc_auc"],
        "Status": "🟢 Active"
    },
    {
        "Module": "Inventory Optimization",
        "Algorithm": "ABC / EOQ Heuristics Engine",
        "Main Metric": "Inventory Health Score",
        "Current Value": inventory_m["health_score"],
        "Status": "🟢 Active"
    }
]

comp_df = pd.DataFrame(comparison_rows)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ==========================================
# SECTION 8 – DEPLOYMENT STATUS
# ==========================================
st.markdown("<div class='section-title'>Section 8: Deployment Status</div>", unsafe_allow_html=True)

deployment_data = [
    {
        "Model / Engine": "Customer Segmentation (KMeans)",
        "Development": "🟢 Completed",
        "Tested": "🟢 Completed",
        "Dashboard Integrated": "🟢 Completed",
        "Production Ready": "🟢 Completed"
    },
    {
        "Model / Engine": "True Churn Prediction (LightGBM)",
        "Development": "🟢 Completed",
        "Tested": "🟢 Completed",
        "Dashboard Integrated": "🟢 Completed",
        "Production Ready": "🟢 Completed"
    },
    {
        "Model / Engine": "Demand Forecasting (XGBoost)",
        "Development": "🟢 Completed",
        "Tested": "🟢 Completed",
        "Dashboard Integrated": "🟢 Completed",
        "Production Ready": "🟢 Completed"
    },
    {
        "Model / Engine": "Inventory Optimization (ABC/EOQ)",
        "Development": "🟢 Completed",
        "Tested": "🟢 Completed",
        "Dashboard Integrated": "🟢 Completed",
        "Production Ready": "🟢 Completed"
    }
]

deploy_df = pd.DataFrame(deployment_data)
st.dataframe(deploy_df, use_container_width=True, hide_index=True)

# ==========================================
# SECTION 9 – BUSINESS PURPOSE
# ==========================================
st.markdown("<div class='section-title'>Section 9: Business Purpose</div>", unsafe_allow_html=True)

purpose_col1, purpose_col2 = st.columns(2)
with purpose_col1:
    st.markdown("""
        <div style="background-color:var(--card-bg); border:1px solid var(--border-color); border-radius:8px; padding:16px; margin-bottom:16px;">
            <h5 style="margin-top:0; color:var(--slate-900);">👥 Customer Segmentation</h5>
            <p style="margin:0; font-size:0.875rem; color:var(--slate-700); line-height:1.5;">
                <strong>"Groups customers with similar purchasing behavior."</strong><br>
                Enables targeted marketing campaigns by segmenting the customer base into VIP, Potential Loyalist, and At-Risk groups. This helps optimize customer lifetime value and acquisition costs.
            </p>
        </div>
        <div style="background-color:var(--card-bg); border:1px solid var(--border-color); border-radius:8px; padding:16px;">
            <h5 style="margin-top:0; color:var(--slate-900);">🎯 True Churn Prediction</h5>
            <p style="margin:0; font-size:0.875rem; color:var(--slate-700); line-height:1.5;">
                <strong>"Predicts customers likely to stop purchasing."</strong><br>
                Identifies high-risk customer accounts before they disengage, allowing proactive customer support and tailored loyalty incentives to preserve retention indices.
            </p>
        </div>
    """, unsafe_allow_html=True)

with purpose_col2:
    st.markdown("""
        <div style="background-color:var(--card-bg); border:1px solid var(--border-color); border-radius:8px; padding:16px; margin-bottom:16px;">
            <h5 style="margin-top:0; color:var(--slate-900);">📈 Demand Forecasting</h5>
            <p style="margin:0; font-size:0.875rem; color:var(--slate-700); line-height:1.5;">
                <strong>"Forecasts future weekly sales for planning."</strong><br>
                Provides an 8-week horizon of projected revenue and holiday demand surges, allowing accurate capacity planning, logistics scheduling, and storage optimization.
            </p>
        </div>
        <div style="background-color:var(--card-bg); border:1px solid var(--border-color); border-radius:8px; padding:16px;">
            <h5 style="margin-top:0; color:var(--slate-900);">📦 Inventory Optimization</h5>
            <p style="margin:0; font-size:0.875rem; color:var(--slate-700); line-height:1.5;">
                <strong>"Optimizes reorder decisions and stock levels."</strong><br>
                Applies Operations Research logic (ABC Analysis, EOQ, Reorder Point) to balance safety stock buffers and product holding costs, preventing costly stockout events.
            </p>
        </div>
    """, unsafe_allow_html=True)
