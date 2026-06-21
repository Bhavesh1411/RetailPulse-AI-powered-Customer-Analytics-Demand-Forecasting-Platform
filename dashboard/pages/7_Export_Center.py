import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# Protect the page
require_auth()

# Render standard sidebar
render_sidebar()

# Inject Custom CSS with 8px grid spacing, CSS variables, and reduced motion settings
st.markdown("""
    <style>
    :root {
        --primary-accent: #2563EB;
        --secondary-accent: #0EA5E9;
        --success: #22C55E;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-color: #F8FAFC;
        --card-bg: #FFFFFF;
        --text-color: #0F172A;
    }
    
    /* Global reduced motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-delay: 0s !important;
            animation-duration: 0s !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0s !important;
            scroll-behavior: auto !important;
        }
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color);
        margin-top: 32px;
        margin-bottom: 16px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
    }
    
    .export-card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
        margin-bottom: 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .export-title {
        color: var(--text-color);
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .export-metadata {
        color: #475569;
        font-size: 0.875rem;
        margin-bottom: 16px;
        line-height: 1.5;
    }
    
    .metadata-item {
        margin-bottom: 4px;
    }
    
    .metadata-label {
        font-weight: 500;
        color: #64748B;
    }
    
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
        width: fit-content;
    }
    
    .status-active {
        background-color: #DEF7EC;
        color: #03543F;
    }
    
    .status-inactive {
        background-color: #FDE8E8;
        color: #9B1C1C;
    }
    
    /* Premium button styles to override defaults */
    div.stDownloadButton > button {
        background-color: var(--primary-accent) !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #1D4ED8 !important;
        color: white !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<header>
    <h1 style="color:#0F172A; margin-bottom: 8px;">📤 Export Center</h1>
    <p style="color:#64748B; font-size:1.1rem; margin-top:0; margin-bottom:24px;">
        Centralized reporting hub. Download generated platform outputs, data tables, and consolidated summaries.
    </p>
</header>
""", unsafe_allow_html=True)

# Resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_dir = os.path.dirname(current_dir)
base_dir = os.path.dirname(dashboard_dir)

seg_path = os.path.join(base_dir, "customer_segmentation", "datasets", "customer_segments_kmeans_finalone.csv")
churn_path = os.path.join(base_dir, "churn_prediction_true", "predictions", "customer_true_churn_predictions.csv")
forecast_path = os.path.join(base_dir, "Demand_Forecasting", "datasets", "weekly_forecast_dashboard_data.json")

inventory_dir = os.path.join(base_dir, "inventory_optimization", "outputs")
inv_master_path = os.path.join(inventory_dir, "inventory_master.csv")
reorder_path = os.path.join(inventory_dir, "reorder_recommendations.csv")
stockout_path = os.path.join(inventory_dir, "stockout_risk_report.csv")
inventory_kpi_path = os.path.join(inventory_dir, "inventory_kpi_summary.csv")

# Helper function to read file metadata
def get_file_info(filepath):
    if not os.path.exists(filepath):
        return {
            "exists": False,
            "filename": os.path.basename(filepath),
            "size": "N/A",
            "last_updated": "N/A",
            "records": "N/A"
        }
    
    filename = os.path.basename(filepath)
    size_bytes = os.path.getsize(filepath)
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.2f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        
    mtime = os.path.getmtime(filepath)
    last_updated = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    records_str = "N/A"
    try:
        if filepath.endswith('.csv'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                row_count = sum(1 for line in f) - 1
                records_str = f"{max(0, row_count):,}"
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    records_str = f"{len(data):,}"
                elif isinstance(data, dict):
                    hist_len = len(data.get('historical', []))
                    fut_len = len(data.get('future', []))
                    if hist_len or fut_len:
                        records_str = f"{hist_len + fut_len:,} (Historical: {hist_len}, Future: {fut_len})"
                    else:
                        records_str = f"{len(data.keys()):,} keys"
        elif filepath.endswith('.md') or filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for line in f)
                records_str = f"{line_count:,} lines"
    except Exception:
        records_str = "Error reading"
        
    return {
        "exists": True,
        "filename": filename,
        "size": size_str,
        "last_updated": last_updated,
        "records": records_str
    }

# Helper to render download cards uniformally
def render_export_card(title, filepath, mime_type, icon="📄", extra_info=None):
    info = get_file_info(filepath)
    status_class = "status-active" if info["exists"] else "status-inactive"
    status_text = "Available" if info["exists"] else "Missing"
    
    extra_html = ""
    if extra_info:
        for k, v in extra_info.items():
            extra_html += f'<div class="metadata-item"><span class="metadata-label">{k}:</span> {v}</div>'
            
    # Build HTML string without blank lines
    html_content = f"""<div class="export-card">
<div>
<div class="export-title">{icon} {title}</div>
<div class="status-badge {status_class}">{status_text}</div>
<div class="export-metadata">
<div class="metadata-item"><span class="metadata-label">File Name:</span> {info['filename']}</div>
<div class="metadata-item"><span class="metadata-label">Record Count:</span> {info['records']}</div>
<div class="metadata-item"><span class="metadata-label">File Size:</span> {info['size']}</div>
<div class="metadata-item"><span class="metadata-label">Last Updated:</span> {info['last_updated']}</div>"""

    if extra_html:
        html_content += f"\n{extra_html}"

    html_content += """
</div>
</div>
</div>"""

    st.markdown(html_content, unsafe_allow_html=True)
    
    if info["exists"]:
        try:
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            st.download_button(
                label=f"Download {title}",
                data=file_bytes,
                file_name=info['filename'],
                mime=mime_type,
                key=f"dl_{info['filename']}"
            )
        except Exception as e:
            st.error(f"Error loading download data: {e}")
    else:
        st.button(f"{title} Unavailable", disabled=True, key=f"disabled_{info['filename']}")

# -----------------
# SECTIONS 1 & 2: CUSTOMER ANALYTICS EXPORTS
# -----------------
st.markdown("<div class='section-header'>Sections 1 & 2: Customer Analytics Exports</div>", unsafe_allow_html=True)
col_c1, col_c2 = st.columns(2)

with col_c1:
    render_export_card(
        title="Customer Segmentation Dataset",
        filepath=seg_path,
        mime_type="text/csv",
        icon="👥"
    )

with col_c2:
    render_export_card(
        title="Churn Prediction Predictions",
        filepath=churn_path,
        mime_type="text/csv",
        icon="🎯"
    )

# -----------------
# SECTION 3: DEMAND FORECASTING EXPORTS
# -----------------
st.markdown("<div class='section-header'>Section 3: Demand Forecasting Exports</div>", unsafe_allow_html=True)
col_f1, col_f2 = st.columns(2)

with col_f1:
    # Load model info from json if exists
    extra_f = None
    if os.path.exists(forecast_path):
        try:
            with open(forecast_path, 'r') as f:
                f_data = json.load(f)
            mi = f_data.get('model_info', {})
            extra_f = {
                "Forecast Model": mi.get('model_name', 'XGBoost'),
                "Validation MAPE": f"{mi.get('val_mape', 0.0):.2f}%",
                "Forecast Horizon": mi.get('horizon', '8 Weeks'),
                "Status": mi.get('status', 'Unknown')
            }
        except Exception:
            pass
            
    render_export_card(
        title="Weekly Forecast Dashboard Data",
        filepath=forecast_path,
        mime_type="application/json",
        icon="📊",
        extra_info=extra_f
    )

with col_f2:
    st.markdown("""<div class="export-card" style="margin-bottom: 12px; padding-bottom: 12px;">
<div class="export-title">📋 Forecasting Reports</div>
<div class="export-metadata">Select and download generated analytical reports and business briefs below.</div>
</div>""", unsafe_allow_html=True)
    
    reports = [
        ("Diagnostic Report", "forecast_diagnostic_report.md"),
        ("Evaluation Report", "forecast_evaluation_report.md"),
        ("Readiness Audit", "forecasting_readiness_audit.md"),
        ("Feasibility Assessment", "weekly_feasibility_assessment.md"),
        ("Business Report", "weekly_forecasting_business_report.md")
    ]
    
    for title, fname in reports:
        r_path = os.path.join(base_dir, "Demand_Forecasting", "reports", fname)
        if os.path.exists(r_path):
            r_info = get_file_info(r_path)
            col_lbl, col_btn = st.columns([2, 1])
            with col_lbl:
                st.markdown(f"**{title}** ({r_info['size']})<br><span style='font-size:0.75rem; color:#64748B;'>Updated: {r_info['last_updated']}</span>", unsafe_allow_html=True)
            with col_btn:
                try:
                    with open(r_path, "rb") as f:
                        r_bytes = f.read()
                    st.download_button(
                        label="Download",
                        data=r_bytes,
                        file_name=fname,
                        mime="text/markdown",
                        key=f"dl_{fname}"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.markdown(f"⚠ **{title}** ({fname}) not found.")

# -----------------
# SECTION 4: INVENTORY OPTIMIZATION EXPORTS
# -----------------
st.markdown("<div class='section-header'>Section 4: Inventory Optimization Exports</div>", unsafe_allow_html=True)
col_i1, col_i2 = st.columns(2)
col_i3, col_i4 = st.columns(2)

with col_i1:
    render_export_card(
        title="Inventory Master",
        filepath=inv_master_path,
        mime_type="text/csv",
        icon="📦"
    )

with col_i2:
    render_export_card(
        title="Reorder Recommendations",
        filepath=reorder_path,
        mime_type="text/csv",
        icon="📈"
    )

with col_i3:
    render_export_card(
        title="Stockout Risk Report",
        filepath=stockout_path,
        mime_type="text/csv",
        icon="⚠️"
    )

with col_i4:
    render_export_card(
        title="Inventory KPI Summary",
        filepath=inventory_kpi_path,
        mime_type="text/csv",
        icon="📉"
    )

# -----------------
# SECTION 5: EXECUTIVE SUMMARY REPORT
# -----------------
st.markdown("<div class='section-header'>Section 5: Executive Summary Report</div>", unsafe_allow_html=True)

# Helper function to compile executive summary data dynamically
def get_executive_summary_data():
    data = {
        'platform_status': {},
        'forecasting': None,
        'segmentation': None,
        'churn': None,
        'inventory': None,
        'alerts': None
    }
    
    seg_exists = os.path.exists(seg_path)
    churn_exists = os.path.exists(churn_path)
    forecast_exists = os.path.exists(forecast_path)
    inv_exists = os.path.exists(inventory_kpi_path)
    
    data['platform_status'] = {
        'Customer Segmentation': 'ACTIVE' if seg_exists else 'INACTIVE',
        'Churn Prediction': 'ACTIVE' if churn_exists else 'INACTIVE',
        'Demand Forecasting': 'ACTIVE' if forecast_exists else 'INACTIVE',
        'Inventory Optimization': 'ACTIVE' if inv_exists else 'INACTIVE'
    }
    
    if forecast_exists:
        try:
            with open(forecast_path, 'r') as f:
                f_data = json.load(f)
            mi = f_data.get("model_info", {})
            f_dict = {
                'model_name': mi.get('model_name', 'XGBoost Regressor'),
                'granularity': mi.get('granularity', 'Weekly'),
                'val_mape': mi.get('val_mape', 0.0),
                'test_mape': mi.get('test_mape', 0.0),
                'horizon': mi.get('horizon', '8 Weeks'),
                'status': mi.get('status', 'Unknown')
            }
            future_list = f_data.get('future', [])
            if future_list:
                f_df = pd.DataFrame(future_list)
                x_idx = np.arange(len(f_df))
                y_val = f_df['forecast'].values
                slope, _ = np.polyfit(x_idx, y_val, 1)
                trend = "Growth" if slope > 1000 else ("Decline" if slope < -1000 else "Stable")
                f_dict['trend'] = trend
                f_dict['slope'] = slope
                f_dict['next_week'] = f_df['forecast'].iloc[0]
                f_dict['peak'] = f_df['forecast'].max()
            data['forecasting'] = f_dict
        except Exception:
            pass
            
    if seg_exists:
        try:
            seg_df = pd.read_csv(seg_path)
            total_customers = len(seg_df)
            total_rev = seg_df['monetary'].sum() if 'monetary' in seg_df.columns else 0.0
            avg_val = total_rev / total_customers if total_customers > 0 else 0.0
            num_seg = seg_df['cluster'].nunique() if 'cluster' in seg_df.columns else 0
            high_val_count = len(seg_df[seg_df['high_value_customer_flag'] == 1]) if 'high_value_customer_flag' in seg_df.columns else 0
            
            data['segmentation'] = {
                'total_customers': total_customers,
                'total_rev': total_rev,
                'avg_val': avg_val,
                'num_seg': num_seg,
                'high_val_count': high_val_count,
                'high_val_pct': (high_val_count/total_customers * 100) if total_customers > 0 else 0
            }
        except Exception:
            pass
            
    if churn_exists:
        try:
            churn_df = pd.read_csv(churn_path)
            total_churn_customers = len(churn_df)
            high_risk_df = churn_df[churn_df['churn_probability'] > 0.70]
            high_risk_count = len(high_risk_df)
            high_risk_pct = (high_risk_count / total_churn_customers) * 100 if total_churn_customers > 0 else 0.0
            rev_at_risk = high_risk_df['monetary'].sum() if 'monetary' in high_risk_df.columns else 0.0
            
            status_text = 'CRITICAL RISK' if high_risk_pct >= 20.0 else ('ATTENTION REQUIRED' if high_risk_pct >= 10.0 else 'HEALTHY')
            data['churn'] = {
                'total_customers': total_churn_customers,
                'high_risk_count': high_risk_count,
                'high_risk_pct': high_risk_pct,
                'rev_at_risk': rev_at_risk,
                'status': status_text
            }
        except Exception:
            pass
            
    if inv_exists:
        try:
            inv_df = pd.read_csv(inventory_kpi_path)
            inv_dict = dict(zip(inv_df['metric'], inv_df['value']))
            
            health_score = float(inv_dict.get('inventory_health_score', 0))
            tot_prod = int(float(inv_dict.get('total_products', 0)))
            crit_risk = int(float(inv_dict.get('risk_category_counts.Critical', 0)))
            sim_val = float(inv_dict.get('total_simulated_inventory_value_usd', 0))
            ex_val = float(inv_dict.get('excess_stock_value_usd', 0))
            
            reorder_count = 0
            if os.path.exists(reorder_path):
                reorder_df = pd.read_csv(reorder_path)
                reorder_count = len(reorder_df)
                
            data['inventory'] = {
                'health_score': health_score,
                'tot_prod': tot_prod,
                'crit_risk': crit_risk,
                'crit_risk_pct': (crit_risk/tot_prod * 100) if tot_prod > 0 else 0,
                'reorder_count': reorder_count,
                'sim_val': sim_val,
                'ex_val': ex_val
            }
        except Exception:
            pass
            
    try:
        c_df = pd.read_csv(churn_path) if churn_exists else pd.DataFrame()
        r_df = pd.read_csv(reorder_path) if os.path.exists(reorder_path) else pd.DataFrame()
        
        with open(forecast_path, 'r') as f:
            d_data = json.load(f) if forecast_exists else {}
            
        t_high_risk = len(c_df[c_df['churn_probability'] > 0.70]) if not c_df.empty else 0
        c_prod = len(r_df[r_df['risk_category'] == 'Critical']) if not r_df.empty else 0
        imm_reorder = len(r_df[r_df['reorder_urgency'] >= 0.80]) if not r_df.empty else 0
        
        d_surges = []
        d_drops = []
        h_peaks = []
        if d_data and 'future' in d_data:
            fut_df = pd.DataFrame(d_data['future'])
            if not fut_df.empty:
                for i, row in fut_df.iterrows():
                    if i == 0:
                        wow = row['growth_from_last_historical']
                    else:
                        wow = (row['forecast'] - fut_df.iloc[i-1]['forecast']) / fut_df.iloc[i-1]['forecast'] * 100
                    if wow > 15:
                        d_surges.append(row)
                    elif wow < -15:
                        d_drops.append(row)
                    if '12-19' in row['date'] or '12-20' in row['date'] or '12-21' in row['date'] or '12-22' in row['date']:
                        h_peaks.append(row)
                        
        crit_alerts = (1 if t_high_risk > 100 else 0) + (1 if c_prod > 50 else 0) + len(d_drops)
        opp_alerts = len(d_surges) + len(h_peaks)
        warn_alerts = (1 if 0 < t_high_risk <= 100 else 0) + (1 if 0 < c_prod <= 50 else 0) + (1 if imm_reorder > 0 else 0)
        tot_alerts = crit_alerts + warn_alerts + opp_alerts
        
        data['alerts'] = {
            'tot_alerts': tot_alerts,
            'crit_alerts': crit_alerts,
            'warn_alerts': warn_alerts,
            'opp_alerts': opp_alerts,
            'imm_reorder': imm_reorder
        }
    except Exception:
        pass
        
    return data

def generate_txt_report(data):
    report_lines = []
    report_lines.append("================================================================================")
    report_lines.append("                     RETAILPULSE EXECUTIVE SUMMARY REPORT")
    report_lines.append(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("================================================================================")
    report_lines.append("")
    
    # 1. Platform Status Summary
    report_lines.append("1. PLATFORM MODULE STATUS")
    report_lines.append("-------------------------")
    for k, v in data['platform_status'].items():
        report_lines.append(f"- {k}: {v}")
    report_lines.append("")
    
    # 2. Demand Forecasting KPIs
    report_lines.append("2. DEMAND FORECASTING OVERVIEW")
    report_lines.append("------------------------------")
    if data['forecasting']:
        f = data['forecasting']
        report_lines.append(f"- Forecast Model:          {f.get('model_name')}")
        report_lines.append(f"- Granularity:             {f.get('granularity')}")
        report_lines.append(f"- Validation MAPE:         {f.get('val_mape'):.2f}%")
        report_lines.append(f"- Test MAPE:               {f.get('test_mape'):.2f}%")
        report_lines.append(f"- Forecast Horizon:        {f.get('horizon')}")
        report_lines.append(f"- Model Status:            {f.get('status')}")
        if 'trend' in f:
            report_lines.append(f"- 8-Week Revenue Trend:    {f.get('trend')} (Slope: {f.get('slope'):+.2f}/week)")
            report_lines.append(f"- Next Week Forecast:      ${f.get('next_week'):,.2f}")
            report_lines.append(f"- Horizon Peak Forecast:   ${f.get('peak'):,.2f}")
    else:
        report_lines.append("Demand forecasting data not available.")
    report_lines.append("")
    
    # 3. Customer Segmentation KPIs
    report_lines.append("3. CUSTOMER SEGMENTATION & REVENUE")
    report_lines.append("----------------------------------")
    if data['segmentation']:
        s = data['segmentation']
        report_lines.append(f"- Total Tracked Customers: {s['total_customers']:,}")
        report_lines.append(f"- Total Platform Revenue:  ${s['total_rev']:,.2f}")
        report_lines.append(f"- Average Customer Value:  ${s['avg_val']:,.2f}")
        report_lines.append(f"- Active Customer Clusters: {s['num_seg']}")
        report_lines.append(f"- High-Value Customers:    {s['high_val_count']:,} ({s['high_val_pct']:.1f}% of total base)")
    else:
        report_lines.append("Customer segmentation data not available.")
    report_lines.append("")
    
    # 4. True Churn Predictions
    report_lines.append("4. CUSTOMER CHURN RISK ANALYSIS")
    report_lines.append("-------------------------------")
    if data['churn']:
        c = data['churn']
        report_lines.append(f"- Evaluated Accounts:      {c['total_customers']:,}")
        report_lines.append(f"- High-Risk Customers:     {c['high_risk_count']:,} ({c['high_risk_pct']:.1f}% of total)")
        report_lines.append(f"- Total Revenue at Risk:   ${c['rev_at_risk']:,.2f}")
        report_lines.append(f"- Churn Platform Status:   {c['status']}")
    else:
        report_lines.append("Churn prediction data not available.")
    report_lines.append("")
    
    # 5. Inventory Optimization
    report_lines.append("5. INVENTORY & SUPPLY CHAIN OPTIMIZATION")
    report_lines.append("----------------------------------------")
    if data['inventory']:
        i = data['inventory']
        report_lines.append(f"- Inventory Health Score:  {i['health_score']:.2f} / 100")
        report_lines.append(f"- Total Products Tracked:  {i['tot_prod']:,}")
        report_lines.append(f"- Critical Risk Products:  {i['crit_risk']:,} ({i['crit_risk_pct']:.1f}% of catalog)")
        report_lines.append(f"- Reorder Recommendations: {i['reorder_count']:,}")
        report_lines.append(f"- Simulated Stock Value:   ${i['sim_val']:,.2f}")
        report_lines.append(f"- Excess Inventory Value:  ${i['ex_val']:,.2f}")
    else:
        report_lines.append("Inventory optimization data not available.")
    report_lines.append("")
    
    # 6. Priority Alerts Count
    report_lines.append("6. SYSTEM ALERTS SUMMARY")
    report_lines.append("------------------------")
    if data['alerts']:
        a = data['alerts']
        report_lines.append(f"- Active Alerts Count:     {a['tot_alerts']}")
        report_lines.append(f"  • Critical Alerts:       {a['crit_alerts']}")
        report_lines.append(f"  • Warning Alerts:        {a['warn_alerts']}")
        report_lines.append(f"  • Opportunity Alerts:    {a['opp_alerts']}")
        report_lines.append(f"  • Immediate Reorders Req: {a['imm_reorder']}")
    else:
        report_lines.append("Alerts data not available.")
        
    report_lines.append("")
    report_lines.append("================================================================================")
    report_lines.append("                        END OF RETAILPULSE EXECUTIVE REPORT")
    report_lines.append("================================================================================")
    
    return "\n".join(report_lines)

# Compile data and text
exec_data = get_executive_summary_data()
report_text = generate_txt_report(exec_data)

def render_dashboard_card(title, kpis, icon=""):
    html = f"""<div class="export-card" style="height: 100%;">
<h3 style="margin-top:0; font-size: 1.1rem; color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">{icon} {title}</h3>
"""
    for label, val in kpis:
        html += f'<div style="display: flex; justify-content: space-between; margin-bottom: 8px;"><span style="color: #64748b; font-size: 0.9rem;">{label}</span><span style="font-weight: 600; color: #0f172a; font-size: 0.95rem;">{val}</span></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_status_badge(status):
    if status == 'ACTIVE' or status == 'HEALTHY':
        return f'<span class="status-badge status-active">{status}</span>'
    elif status == 'INACTIVE':
        return f'<span class="status-badge status-inactive">{status}</span>'
    elif status == 'CRITICAL RISK':
        return f'<span class="status-badge status-inactive" style="background-color: #fef2f2; color: #ef4444;">{status}</span>'
    else:
        return f'<span class="status-badge" style="background-color: #fef9c3; color: #854d0e;">{status}</span>'

# Dashboard Layout Row 1
col1, col2, col3 = st.columns(3)

with col1:
    kpis = [(mod, render_status_badge(stat)) for mod, stat in exec_data['platform_status'].items()]
    render_dashboard_card("Platform Status", kpis, "🟢")

with col2:
    if exec_data['forecasting']:
        f = exec_data['forecasting']
        kpis = [
            ("Model", f.get('model_name')),
            ("Val MAPE", f"{f.get('val_mape'):.2f}%"),
            ("Horizon", f.get('horizon')),
            ("Revenue Trend", f.get('trend', 'N/A'))
        ]
    else:
        kpis = [("Status", "Data Unavailable")]
    render_dashboard_card("Demand Forecasting", kpis, "📈")

with col3:
    if exec_data['segmentation']:
        s = exec_data['segmentation']
        kpis = [
            ("Total Customers", f"{s['total_customers']:,}"),
            ("Platform Revenue", f"${s['total_rev']:,.2f}"),
            ("High-Value Base", f"{s['high_val_pct']:.1f}%"),
            ("Active Clusters", f"{s['num_seg']}")
        ]
    else:
        kpis = [("Status", "Data Unavailable")]
    render_dashboard_card("Customer Segmentation", kpis, "👥")

# Dashboard Layout Row 2
col4, col5, col6 = st.columns(3)

with col4:
    if exec_data['churn']:
        c = exec_data['churn']
        kpis = [
            ("Platform Status", render_status_badge(c['status'])),
            ("High-Risk Count", f"{c['high_risk_count']:,}"),
            ("Revenue at Risk", f"${c['rev_at_risk']:,.2f}")
        ]
    else:
        kpis = [("Status", "Data Unavailable")]
    render_dashboard_card("Churn Prediction", kpis, "⚠️")

with col5:
    if exec_data['inventory']:
        i = exec_data['inventory']
        kpis = [
            ("Health Score", f"{i['health_score']:.1f} / 100"),
            ("Critical Risk Items", f"{i['crit_risk']:,}"),
            ("Reorders Req", f"{i['reorder_count']:,}"),
            ("Excess Stock Value", f"${i['ex_val']:,.2f}")
        ]
    else:
        kpis = [("Status", "Data Unavailable")]
    render_dashboard_card("Inventory Optimization", kpis, "📦")

with col6:
    if exec_data['alerts']:
        a = exec_data['alerts']
        kpis = [
            ("Total Active Alerts", f"<span style='font-weight: 700; color: #0f172a;'>{a['tot_alerts']}</span>"),
            ("Critical Alerts", f"<span style='color: #ef4444;'>{a['crit_alerts']}</span>"),
            ("Opportunity Alerts", f"<span style='color: #22c55e;'>{a['opp_alerts']}</span>"),
            ("Immediate Reorders", f"<span style='color: #f59e0b;'>{a['imm_reorder']}</span>")
        ]
    else:
        kpis = [("Status", "Data Unavailable")]
    render_dashboard_card("System Alerts", kpis, "🔔")

st.markdown("<br>", unsafe_allow_html=True)

# Export as TXT file
st.download_button(
    label="Download Executive Summary Report (TXT)",
    data=report_text,
    file_name="retailpulse_executive_summary.txt",
    mime="text/plain",
    key="dl_executive_summary"
)

