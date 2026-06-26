import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from auth.session_manager import require_auth
from components.sidebar import render_sidebar

# Protect the page
require_auth()

# Render standard sidebar
render_sidebar()

# Inject CSS adhering to RetailPulse Light Theme and global rules (8px grid, variables, reduced motion)
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
    
    /* Spacing & Layout (8px grid) */
    .dashboard-container {
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
    
    /* Metric Cards Grid */
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
    
    /* Banner styles for merge validation */
    .validation-banner {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 24px;
    }
    
    .validation-title {
        font-weight: 600;
        font-size: 1rem;
        color: var(--primary);
        margin-bottom: 8px;
    }
    
    .validation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
    }
    
    .validation-item {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
    
    .validation-num {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--slate-900);
    }
    
    .validation-label {
        font-size: 0.75rem;
        color: var(--slate-500);
        margin-top: 4px;
    }
    
    /* Status Badge styling */
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
    </style>
""", unsafe_allow_html=True)

# Resolve Base Paths
current_dir = os.path.dirname(os.path.abspath(__file__)) 
dashboard_dir = os.path.dirname(current_dir)
base_dir = os.path.dirname(dashboard_dir)

# ==========================================
# DATA LOADING & MERGING
# ==========================================

@st.cache_data
def load_all_datasets():
    """
    Loads all relevant prediction datasets and performs the in-memory outer join 
    with merge validation stats computed dynamically.
    """
    seg_path = os.path.join(base_dir, "customer_segmentation", "datasets", "customer_segments_kmeans_finalone.csv")
    churn_path = os.path.join(base_dir, "churn_prediction_true", "predictions", "customer_true_churn_predictions.csv")
    inventory_path = os.path.join(base_dir, "inventory_optimization", "outputs", "inventory_master.csv")
    forecast_path = os.path.join(base_dir, "Demand_Forecasting", "datasets", "weekly_forecast_dashboard_data.json")
    
    df_merged = pd.DataFrame()
    validation_metrics = {
        "total_seg": 0,
        "total_churn": 0,
        "matched_count": 0,
        "unmatched_count": 0,
        "unmatched_ids": [],
        "success": False
    }
    
    # 1. Load Customer Datasets & Perform Join
    if os.path.exists(seg_path) and os.path.exists(churn_path):
        df_seg = pd.read_csv(seg_path)
        df_seg['customer_id'] = df_seg['customer_id'].astype(str).str.strip()
        
        df_churn = pd.read_csv(churn_path)
        df_churn['customer_id'] = df_churn['customer_id'].astype(str).str.strip()
        
        # Calculate overlap details
        seg_ids = set(df_seg['customer_id'])
        churn_ids = set(df_churn['customer_id'])
        
        matched_ids = seg_ids.intersection(churn_ids)
        unmatched_ids = seg_ids.symmetric_difference(churn_ids)
        
        validation_metrics = {
            "total_seg": len(df_seg),
            "total_churn": len(df_churn),
            "matched_count": len(matched_ids),
            "unmatched_count": len(unmatched_ids),
            "unmatched_ids": sorted(list(unmatched_ids)),
            "success": True
        }
        
        # Outer merge to retain unmatched records
        df_merged = pd.merge(df_seg, df_churn, on='customer_id', how='outer', suffixes=('_seg', '_churn'))
        
        # Combine overlapping attributes (take segmentation first, fallback to churn)
        for col in ['recency', 'frequency', 'monetary', 'customer_loyalty_score']:
            seg_col = f"{col}_seg"
            churn_col = f"{col}_churn"
            if seg_col in df_merged.columns and churn_col in df_merged.columns:
                df_merged[col] = df_merged[seg_col].fillna(df_merged[churn_col])
                df_merged.drop(columns=[seg_col, churn_col], inplace=True)
    else:
        st.warning("Customer Segmentation or Churn datasets not found. Validation statistics unavailable.")
        
    # 2. Load Inventory Master
    df_inventory = pd.DataFrame()
    if os.path.exists(inventory_path):
        df_inventory = pd.read_csv(inventory_path)
        df_inventory['product_id'] = df_inventory['product_id'].astype(str).str.strip()
    else:
        st.warning("Inventory Optimization master dataset not found.")
        
    # 3. Load Weekly Forecast JSON
    forecast_json = {}
    if os.path.exists(forecast_path):
        with open(forecast_path, 'r', encoding='utf-8') as f:
            forecast_json = json.load(f)
    else:
        st.warning("Weekly Sales Forecast JSON dataset not found.")
        
    return df_merged, validation_metrics, df_inventory, forecast_json

df_merged, validation_metrics, df_inventory, forecast_json = load_all_datasets()

# ==========================================
# RECOMMENDATION ENGINE HELPER FUNCTIONS (Modification 2)
# ==========================================

def get_customer_recommendation(customer_row):
    """
    Decoupled helper function returning business recommendations based on
    customer segmentation, churn predictions, and past purchases.
    """
    cluster = customer_row.get('cluster')
    churn_prob = customer_row.get('churn_probability')
    risk_cat = customer_row.get('risk_category')
    high_value = customer_row.get('high_value_customer_flag')
    recency = customer_row.get('recency')
    monetary = customer_row.get('monetary')

    # Convert numeric fields safely
    try:
        churn_prob = float(churn_prob) if not pd.isna(churn_prob) else None
    except:
        churn_prob = None
        
    try:
        high_value = int(high_value) if not pd.isna(high_value) else 0
    except:
        high_value = 0

    try:
        recency = float(recency) if not pd.isna(recency) else None
    except:
        recency = None

    try:
        monetary = float(monetary) if not pd.isna(monetary) else None
    except:
        monetary = None

    recs = []
    
    # 1. Critical Churn Intervention (High Risk Category)
    if risk_cat == 'High Risk' or (churn_prob is not None and churn_prob >= 0.7):
        recs.append({
            "action": "Urgent Loyalty Retention Offer",
            "details": "Customer shows critical signs of churn. Initiate personal email outreach or assign an account executive to offer a tailored discount incentive (e.g., 20% off next purchase) immediately.",
            "priority": "Critical"
        })
    elif risk_cat == 'Medium Risk' or (churn_prob is not None and churn_prob >= 0.4):
        recs.append({
            "action": "Proactive Re-engagement outreach",
            "details": "Customer is showing signs of disengagement. Deploy automated customized product recommendation suggestions with a small promo code.",
            "priority": "High"
        })

    # 2. VIP Recognition (Cluster 0 or High Value Customer Flag)
    if high_value == 1 or cluster == 'Cluster 0' or (monetary is not None and monetary > 2000):
        recs.append({
            "action": "Premium VIP Loyalty Club Invitation",
            "details": "Customer belongs to a high-value or VIP tier. Enroll them in early access product notifications, waive delivery fees, and assign priority customer support routing.",
            "priority": "High"
        })

    # 3. Growth Cross-selling / Up-selling (Low risk and active)
    if (risk_cat == 'Low Risk' or (churn_prob is not None and churn_prob < 0.4)) and (recency is not None and recency <= 45):
        recs.append({
            "action": "Cross-sell Top Revenue Lines",
            "details": "Active customer with stable retention index. Cross-sell complementary high-revenue items from inventory ABC Class A catalogs.",
            "priority": "Standard"
        })

    # 4. Inactive Cohort Win-Back (Inactive over 90 days but high historical monetary values)
    if recency is not None and recency > 90:
        recs.append({
            "action": "Win-back Re-activation Campaign",
            "details": "Customer hasn't placed an order in over 90 days. Deliver a 'We miss you' voucher with direct communication to re-activate the account.",
            "priority": "High"
        })

    # Fallback recommendations if no specific logic applies
    if not recs:
        recs.append({
            "action": "Standard Nurture Stream",
            "details": "Include in regular customer newsletter distributions and track transaction changes on a monthly basis.",
            "priority": "Low"
        })

    return recs


def get_inventory_recommendation(product_row):
    """
    Decoupled helper function returning inventory replenishment and reordering 
    strategies based on current stock, demand rates, lead times, and stockout probabilities.
    """
    abc_class = product_row.get('abc_class')
    stockout_prob = product_row.get('stockout_probability')
    urgency = product_row.get('reorder_urgency')
    stock = product_row.get('simulated_current_stock')
    rop = product_row.get('reorder_point')
    eoq = product_row.get('eoq')
    days_to_stockout = product_row.get('days_to_stockout')

    # Convert numeric fields safely
    try:
        stockout_prob = float(stockout_prob) if not pd.isna(stockout_prob) else 0.0
    except:
        stockout_prob = 0.0

    try:
        urgency = float(urgency) if not pd.isna(urgency) else 0.0
    except:
        urgency = 0.0

    try:
        stock = float(stock) if not pd.isna(stock) else 0.0
    except:
        stock = 0.0

    try:
        rop = float(rop) if not pd.isna(rop) else 0.0
    except:
        rop = 0.0

    try:
        eoq = float(eoq) if not pd.isna(eoq) else 0.0
    except:
        eoq = 0.0

    try:
        days_to_stockout = float(days_to_stockout) if not pd.isna(days_to_stockout) else 999.0
    except:
        days_to_stockout = 999.0

    recs = []

    # 1. Critical Inventory Stockout Warnings (Low days-to-stockout or high urgency)
    if stockout_prob >= 0.5 or urgency >= 0.6 or stock < (rop * 0.5):
        recs.append({
            "action": "Immediate Replenishment Triggered",
            "details": f"Stock levels are critically low ({stock:.1f} vs reorder point {rop:.1f}). Place a priority replenishment order of Economic Order Quantity (EOQ = {eoq:.1f} units) immediately.",
            "priority": "Critical"
        })
    elif stock < rop:
        recs.append({
            "action": "Trigger Replenishment Reorder Point",
            "details": f"Current stock ({stock:.1f}) has fallen below the calculated reorder point ({rop:.1f}). Place a standard order of EOQ ({eoq:.1f} units).",
            "priority": "High"
        })

    # 2. Supply Chain Lead Time Acceleration
    if urgency >= 0.8 or (days_to_stockout < 5 and stockout_prob >= 0.8):
        recs.append({
            "action": "Vendor Expedited Freight Request",
            "details": "Estimated stockout date is less than lead time metrics. Coordinate with supplier to accelerate shipping for the replenishment consignment.",
            "priority": "Critical"
        })

    # 3. ABC-based controls
    if abc_class == 'A':
        recs.append({
            "action": "ABC Class A: Strict Oversight",
            "details": "This high-revenue product accounts for top sales velocity. Manage safety stock dynamically and perform rolling weekly inventory counts.",
            "priority": "High"
        })
    elif abc_class == 'C':
        recs.append({
            "action": "ABC Class C: Bulk Ordering Strategy",
            "details": "Low-revenue item. Maintain high safety stock variables to minimize logistics transaction costs and ordering frequency.",
            "priority": "Low"
        })

    # 4. Overstock check
    if rop > 0 and stock > (rop * 3.0):
        recs.append({
            "action": "Inventory Overstock Alert",
            "details": f"Current inventory level ({stock:.1f}) exceeds reorder target buffers significantly. Review promotional options to increase sales velocity and free shelf space.",
            "priority": "Medium"
        })

    if not recs:
        recs.append({
            "action": "Healthy Inventory Levels",
            "details": "Product inventory levels conform to safe buffer targets. Continue automated background tracking.",
            "priority": "Low"
        })

    return recs

# ==========================================
# PAGE RENDER - HEADER
# ==========================================
st.markdown("<h1>🔮 Prediction & Decision Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:var(--slate-700); font-size:1.1rem; margin-top:-0.5rem;'>Decision Support System incorporating customer profiles, churn indices, inventory statuses, and demand projections.</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# MODIFICATION 1 – CUSTOMER MERGE VALIDATION BANNER
# ==========================================
if validation_metrics.get("success", False):
    st.markdown(f"""
        <div class="validation-banner">
            <div class="validation-title">⚙️ Customer Merge Validation Log</div>
            <div class="validation-grid">
                <div class="validation-item">
                    <div class="validation-num">{validation_metrics['total_seg']:,}</div>
                    <div class="validation-label">Customer Segmentation Rows</div>
                </div>
                <div class="validation-item">
                    <div class="validation-num">{validation_metrics['total_churn']:,}</div>
                    <div class="validation-label">True Churn Dataset Rows</div>
                </div>
                <div class="validation-item" style="border-color: #BBF7D0;">
                    <div class="validation-num" style="color: var(--success);">{validation_metrics['matched_count']:,}</div>
                    <div class="validation-label">Successfully Matched IDs</div>
                </div>
                <div class="validation-item" style="border-color: #FECACA;">
                    <div class="validation-num" style="color: var(--danger);">{validation_metrics['unmatched_count']:,}</div>
                    <div class="validation-label">Unmatched IDs (Retained)</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Expandable details for audit log
    with st.expander("🔍 View Details / Unmatched Customer IDs Audit Log"):
        st.markdown(f"**Total Unmatched IDs:** `{validation_metrics['unmatched_count']}`")
        if validation_metrics["unmatched_ids"]:
            st.write(validation_metrics["unmatched_ids"])
        else:
            st.write("All Customer IDs are matched between datasets.")
else:
    st.info("Merge validation skipped: Segmentation and Churn outputs were not found simultaneously.")

# ==========================================
# TAB NAVIGATION
# ==========================================
tab_cust, tab_prod, tab_fore, tab_exec = st.tabs([
    "👤 Customer Decision Support",
    "📦 Inventory replenishment",
    "📈 Demand Forecast Center",
    "🏛️ Executive Decision Summary"
])

# Format converters for UI consistency
def format_val(val, fmt_func=None):
    if pd.isna(val) or val is None or val == "Not Available":
        return "Not Available"
    if fmt_func:
        try:
            return fmt_func(val)
        except:
            return "Not Available"
    return str(val)

# ==========================================
# TAB 1: CUSTOMER DECISION SUPPORT
# ==========================================
with tab_cust:
    st.subheader("Customer Intelligence Lookup & Targeted Campaigns")
    st.markdown("Query integrated segmentation features and retention predictions to trigger recommendations.")
    
    if not df_merged.empty:
        # Build search labels dynamically indicating where customer was matched
        customer_ids = sorted(df_merged['customer_id'].unique().tolist())
        
        # Helper to format option strings for search selectbox
        def get_select_label(cust_id):
            row = df_merged[df_merged['customer_id'] == cust_id].iloc[0]
            in_seg = not pd.isna(row.get('cluster'))
            in_churn = not pd.isna(row.get('risk_category'))
            if in_seg and in_churn:
                return f"Customer ID: {cust_id} [Matched]"
            elif in_seg:
                return f"Customer ID: {cust_id} [Segmentation Only]"
            else:
                return f"Customer ID: {cust_id} [Churn Predictions Only]"
                
        selected_cust_id = st.selectbox(
            "Search and Select Customer ID",
            options=customer_ids,
            format_func=get_select_label,
            key="cust_select_box"
        )
        
        if selected_cust_id:
            cust_row = df_merged[df_merged['customer_id'] == selected_cust_id].iloc[0].to_dict()
            
            # Display matched source banner
            in_seg = not pd.isna(cust_row.get('cluster'))
            in_churn = not pd.isna(cust_row.get('risk_category'))
            if in_seg and in_churn:
                st.success("Selected customer profile is fully matched across customer segmentation & churn forecasting tables.")
            elif in_seg:
                st.warning("Selected customer profile exists only in Customer Segmentation table. Churn characteristics are unavailable.")
            else:
                st.warning("Selected customer profile exists only in Churn Predictions table. Segmentation clusters are unavailable.")
            
            # Columns layout
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.markdown("#### Customer Attributes Profile")
                
                # HTML card layout inside columns
                st.markdown(f"""
                    <div class="metric-grid">
                        <div class="kpi-card">
                            <div class="kpi-title">Cluster Cohort</div>
                            <div class="kpi-value" style="color: var(--primary);">{format_val(cust_row.get('cluster'))}</div>
                            <div class="kpi-subtitle">Customer Segmentation</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Churn Probability</div>
                            <div class="kpi-value">{format_val(cust_row.get('churn_probability'), lambda x: f"{float(x)*100:.2f}%")}</div>
                            <div class="kpi-subtitle">Risk Probability Index</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Churn Risk Category</div>
                            <div class="kpi-value">{format_val(cust_row.get('risk_category'))}</div>
                            <div class="kpi-subtitle">Risk Classification</div>
                        </div>
                    </div>
                    <div class="metric-grid">
                        <div class="kpi-card">
                            <div class="kpi-title">Monetary Value</div>
                            <div class="kpi-value" style="color: var(--success);">{format_val(cust_row.get('monetary'), lambda x: f"${float(x):,.2f}")}</div>
                            <div class="kpi-subtitle">Total Historical Value</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Frequency (Orders)</div>
                            <div class="kpi-value">{format_val(cust_row.get('frequency'), lambda x: f"{int(float(x)):,}")}</div>
                            <div class="kpi-subtitle">Total Order Placements</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Recency Index</div>
                            <div class="kpi-value">{format_val(cust_row.get('recency'), lambda x: f"{int(float(x))} days")}</div>
                            <div class="kpi-subtitle">Days Since Last Order</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display table of secondary attributes
                st.markdown("#### Secondary Attributes Profile")
                sec_df = pd.DataFrame([
                    {"Attribute": "Loyalty Score", "Value": format_val(cust_row.get('customer_loyalty_score'), lambda x: f"{float(x):.2f}/10.00")},
                    {"Attribute": "High Value Flag", "Value": format_val(cust_row.get('high_value_customer_flag'), lambda x: "Yes" if int(float(x)) == 1 else "No")},
                    {"Attribute": "Customer Tenure", "Value": format_val(cust_row.get('customer_tenure'), lambda x: f"{int(float(x))} days")},
                    {"Attribute": "Weekend Purchase Ratio", "Value": format_val(cust_row.get('weekend_sales_ratio'), lambda x: f"{float(x)*100:.2f}%")},
                    {"Attribute": "Revenue Rank", "Value": format_val(cust_row.get('customer_rank_by_revenue'), lambda x: f"#{int(float(x))}")},
                    {"Attribute": "Churn Warning Flag", "Value": format_val(cust_row.get('churn_warning_flag'), lambda x: "Active" if int(float(x)) == 1 else "Inactive")}
                ])
                st.dataframe(sec_df, use_container_width=True, hide_index=True)
                
            with col_right:
                st.markdown("#### Actionable Retention Recommendations")
                st.markdown("Suggested interventions based on customer behavior indices:")
                
                recs = get_customer_recommendation(cust_row)
                for r in recs:
                    priority = r.get("priority", "Standard").lower()
                    badge_color = "var(--primary)"
                    bg_color = "#EFF6FF"
                    if priority == "critical":
                        badge_color = "var(--danger)"
                        bg_color = "#FEF2F2"
                    elif priority == "high":
                        badge_color = "var(--warning)"
                        bg_color = "#FFFBEB"
                    elif priority == "low":
                        badge_color = "var(--slate-500)"
                        bg_color = "#F8FAFC"
                        
                    st.markdown(f"""
                        <div style="
                            border-left: 4px solid {badge_color};
                            background-color: {bg_color};
                            border-radius: 8px;
                            padding: 16px;
                            margin-bottom: 12px;
                            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                        ">
                            <div style="font-weight: 700; font-size: 0.95rem; color: var(--slate-900); display: flex; justify-content: space-between; align-items: center;">
                                <span>🎯 {r.get('action')}</span>
                                <span style="
                                    font-size: 0.75rem; 
                                    padding: 2px 6px; 
                                    border-radius: 4px; 
                                    background-color: {badge_color}; 
                                    color: #FFFFFF;
                                    text-transform: uppercase;
                                    letter-spacing: 0.05em;
                                ">{priority}</span>
                            </div>
                            <p style="margin: 8px 0 0 0; font-size: 0.85rem; color: var(--slate-700); line-height: 1.4;">
                                {r.get('details')}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Customer datasets are not loaded correctly.")

# ==========================================
# TAB 2: INVENTORY DECISION SUPPORT
# ==========================================
with tab_prod:
    st.subheader("Product Replenishment & Stockout Prevention Center")
    st.markdown("Search a Product ID to fetch reorder point parameters and simulated stock alerts.")
    
    if not df_inventory.empty:
        prod_ids = sorted(df_inventory['product_id'].unique().tolist())
        selected_prod_id = st.selectbox(
            "Search and Select Product ID",
            options=prod_ids,
            key="prod_select_box"
        )
        
        if selected_prod_id:
            prod_row = df_inventory[df_inventory['product_id'] == selected_prod_id].iloc[0].to_dict()
            
            # Risk warning label
            risk_category = prod_row.get('risk_category', 'Low')
            if risk_category in ['Critical', 'High']:
                st.error(f"Stockout risk warning is active for this product: {risk_category} Risk.")
            elif risk_category == 'Medium':
                st.warning("Inventory levels are approaching boundaries: Medium Risk.")
            else:
                st.success("Inventory parameters are stable: Low Risk.")
                
            col_l, col_r = st.columns([2, 1])
            
            with col_l:
                st.markdown("#### Inventory Parameters Profile")
                
                # HTML card layout inside columns
                st.markdown(f"""
                    <div class="metric-grid">
                        <div class="kpi-card">
                            <div class="kpi-title">Current Stock Level</div>
                            <div class="kpi-value" style="color: var(--primary);">{format_val(prod_row.get('simulated_current_stock'), lambda x: f"{float(x):.1f} units")}</div>
                            <div class="kpi-subtitle">Simulated Live Units</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Reorder Point (ROP)</div>
                            <div class="kpi-value">{format_val(prod_row.get('reorder_point'), lambda x: f"{float(x):.1f} units")}</div>
                            <div class="kpi-subtitle">Trigger Threshold Buffer</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Economic Order Qty (EOQ)</div>
                            <div class="kpi-value">{format_val(prod_row.get('eoq'), lambda x: f"{float(x):.1f} units")}</div>
                            <div class="kpi-subtitle">Optimized Purchase Volume</div>
                        </div>
                    </div>
                    <div class="metric-grid">
                        <div class="kpi-card">
                            <div class="kpi-title">Stockout Probability</div>
                            <div class="kpi-value" style="color: var(--danger);">{format_val(prod_row.get('stockout_probability'), lambda x: f"{float(x)*100:.2f}%")}</div>
                            <div class="kpi-subtitle">Probability of Depletion</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Reorder Urgency</div>
                            <div class="kpi-value">{format_val(prod_row.get('reorder_urgency'), lambda x: f"{float(x)*100:.2f}%")}</div>
                            <div class="kpi-subtitle">Urgency Index</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-title">Days to Stockout</div>
                            <div class="kpi-value">{format_val(prod_row.get('days_to_stockout'), lambda x: f"{float(x):.1f} days")}</div>
                            <div class="kpi-subtitle">Depletion Forecast Time</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Secondary Control Specifications")
                sec_prod_df = pd.DataFrame([
                    {"Attribute": "ABC Revenue Class", "Value": format_val(prod_row.get('abc_class'))},
                    {"Attribute": "Target Service Level", "Value": format_val(prod_row.get('service_level'), lambda x: f"{float(x)*100:.2f}%")},
                    {"Attribute": "Safety Stock Buffer", "Value": format_val(prod_row.get('safety_stock'), lambda x: f"{float(x):.1f} units")},
                    {"Attribute": "Vendor Lead Time", "Value": format_val(prod_row.get('lead_time'), lambda x: f"{int(float(x))} days")},
                    {"Attribute": "Annual Demand Projection", "Value": format_val(prod_row.get('annual_demand'), lambda x: f"{float(x):,.1f} units")},
                    {"Attribute": "Unit Cost Pricing", "Value": format_val(prod_row.get('unit_cost'), lambda x: f"${float(x):,.2f}")}
                ])
                st.dataframe(sec_prod_df, use_container_width=True, hide_index=True)
                
            with col_r:
                st.markdown("#### Supply Chain Recommendations")
                st.markdown("Automated decisions to balance holding cost vs stockout risks:")
                
                prod_recs = get_inventory_recommendation(prod_row)
                for pr in prod_recs:
                    priority = pr.get("priority", "Standard").lower()
                    badge_color = "var(--primary)"
                    bg_color = "#EFF6FF"
                    if priority == "critical":
                        badge_color = "var(--danger)"
                        bg_color = "#FEF2F2"
                    elif priority == "high":
                        badge_color = "var(--warning)"
                        bg_color = "#FFFBEB"
                    elif priority == "low":
                        badge_color = "var(--slate-500)"
                        bg_color = "#F8FAFC"
                        
                    st.markdown(f"""
                        <div style="
                            border-left: 4px solid {badge_color};
                            background-color: {bg_color};
                            border-radius: 8px;
                            padding: 16px;
                            margin-bottom: 12px;
                            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                        ">
                            <div style="font-weight: 700; font-size: 0.95rem; color: var(--slate-900); display: flex; justify-content: space-between; align-items: center;">
                                <span>📦 {pr.get('action')}</span>
                                <span style="
                                    font-size: 0.75rem; 
                                    padding: 2px 6px; 
                                    border-radius: 4px; 
                                    background-color: {badge_color}; 
                                    color: #FFFFFF;
                                    text-transform: uppercase;
                                    letter-spacing: 0.05em;
                                ">{priority}</span>
                            </div>
                            <p style="margin: 8px 0 0 0; font-size: 0.85rem; color: var(--slate-700); line-height: 1.4;">
                                {pr.get('details')}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Inventory dataset is not loaded correctly.")

# ==========================================
# TAB 3: DEMAND FORECAST CENTER
# ==========================================
with tab_fore:
    st.subheader("Weekly Sales Demand Projections")
    st.markdown("Macro-level projections for sales and holiday anomalies across the next 8 weeks.")
    
    if forecast_json:
        # Extract model info
        model_info = forecast_json.get("model_info", {})
        
        # Display model specifications
        st.markdown(f"""
            <div style="
                background-color: var(--slate-100);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 24px;
            ">
                <h5 style="margin-top:0; color:var(--slate-900);">🤖 Forecasting Model Spec</h5>
                <p style="margin: 0; font-size:0.9rem;">
                    <strong>Algorithm:</strong> {model_info.get('model_name', 'Not Available')} &nbsp;|&nbsp;
                    <strong>Validation MAPE:</strong> {model_info.get('val_mape', 'Not Available'):.2f}% &nbsp;|&nbsp;
                    <strong>Horizon:</strong> {model_info.get('horizon', 'Not Available')} &nbsp;|&nbsp;
                    <strong>Status:</strong> <span class="health-badge badge-active">{model_info.get('status', 'Active')}</span>
                </p>
                <p style="margin: 8px 0 0 0; font-size:0.8rem; color:var(--slate-500); font-style:italic;">
                    {model_info.get('justification', '')}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Process and combine forecasts for chart plotting
        hist_df = pd.DataFrame(forecast_json.get("historical", []))
        fut_df = pd.DataFrame(forecast_json.get("future", []))
        
        if not hist_df.empty and not fut_df.empty:
            chart_rows = []
            
            # Include recent historical points to keep the chart clean
            hist_subset = hist_df.tail(16)  # Last 16 historical weeks
            for idx, r in hist_subset.iterrows():
                chart_rows.append({
                    "Date": r["date"],
                    "Historical Sales": float(r["actual"]) if r["actual"] > 0 else np.nan,
                    "Historical Predictions": float(r["predicted"]) if not pd.isna(r["predicted"]) else np.nan,
                    "Future Forecast": np.nan
                })
                
            # Add future points
            for idx, r in fut_df.iterrows():
                chart_rows.append({
                    "Date": r["date"],
                    "Historical Sales": np.nan,
                    "Historical Predictions": np.nan,
                    "Future Forecast": float(r["forecast"])
                })
                
            chart_df = pd.DataFrame(chart_rows)
            chart_df.set_index("Date", inplace=True)
            
            # Plot
            st.markdown("#### Historical Sales vs Future Horizon Projection")
            st.line_chart(chart_df, height=350, use_container_width=True)
            
            # Forecast breakdown summary cards
            total_future_sales = fut_df["forecast"].sum()
            avg_future_sales = fut_df["forecast"].mean()
            peak_forecast = fut_df["forecast"].max()
            peak_date = fut_df.loc[fut_df["forecast"].idxmax(), "date"]
            
            st.markdown(f"""
                <div class="metric-grid">
                    <div class="kpi-card">
                        <div class="kpi-title">8-Week Projected Revenue</div>
                        <div class="kpi-value" style="color: var(--primary);">${total_future_sales:,.2f}</div>
                        <div class="kpi-subtitle">Accumulated Projection</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Average Weekly Sales</div>
                        <div class="kpi-value">${avg_future_sales:,.2f}</div>
                        <div class="kpi-subtitle">Estimated Mean Sales</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Peak Revenue Week</div>
                        <div class="kpi-value" style="color: var(--success);">${peak_forecast:,.2f}</div>
                        <div class="kpi-subtitle">Projected for: {peak_date}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Projections list table
            st.markdown("#### Upcoming Forecast Horizon Breakdown")
            table_rows = []
            for idx, r in fut_df.iterrows():
                table_rows.append({
                    "Target Date": r["date"],
                    "Projected Revenue": f"${r['forecast']:,.2f}",
                    "Growth Trend (vs Last Actual)": f"{r.get('growth_from_last_historical', 0):+.2f}%"
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Demand forecasting details are not loaded correctly.")

# ==========================================
# TAB 4: EXECUTIVE DECISION SUMMARY
# ==========================================
with tab_exec:
    st.subheader("Executive Overview & Decision Summaries")
    st.markdown("Dynamically compiled overview of platform risks, reorder metrics, and demand horizons.")
    
    # Compute Aggregations
    # 1. Total High Risk Churn Customers
    num_high_risk_customers = 0
    if not df_merged.empty:
        # Check risk_category or churn_probability
        num_high_risk_customers = len(df_merged[
            (df_merged['risk_category'] == 'High Risk') | 
            (df_merged['churn_probability'] >= 0.7)
        ])
        
    # 2. Total Critical Inventory Products
    num_critical_products = 0
    total_products_count = 0
    if not df_inventory.empty:
        total_products_count = len(df_inventory)
        num_critical_products = len(df_inventory[
            (df_inventory['risk_category'].isin(['Critical', 'High'])) | 
            (df_inventory['stockout_probability'] >= 0.5) | 
            (df_inventory['reorder_urgency'] >= 0.6)
        ])
        
    # 3. Forecast Growth Trend Direction
    trend_description = "stable"
    if forecast_json and "future" in forecast_json:
        future_list = forecast_json["future"]
        if len(future_list) >= 4:
            first_half = [x["forecast"] for x in future_list[:4]]
            second_half = [x["forecast"] for x in future_list[-4:]]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first * 1.05:
                trend_description = "upward, indicating growth in subsequent week cycles"
            elif avg_second < avg_first * 0.95:
                trend_description = "downward, indicating contractions in subsequent week cycles"
            else:
                trend_description = "relatively flat and stable"
                
    st.markdown(f"""
        <div style="
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        ">
            <h4 style="margin-top:0; color:var(--slate-900); border-bottom: 1px solid var(--border-color); padding-bottom: 8px;">
                🏛️ Platform Executive Briefing
            </h4>
            <p style="font-size:1.05rem; color:var(--slate-700); line-height: 1.6; margin-top: 16px;">
                Currently, <strong>{num_high_risk_customers}</strong> customers are at high risk of churning, and 
                <strong>{num_critical_products}</strong> products (out of {total_products_count} active catalog codes) 
                require immediate replenishment intervention to prevent localized stockout disruptions. 
                The weekly sales forecast for the next 8 weeks is trending <strong>{trend_description}</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Executive Recommendations Grid
    st.markdown("#### Strategic Actions Priority Grid")
    
    exec_col1, exec_col2 = st.columns(2)
    
    with exec_col1:
        st.markdown("""
            <div style="background-color: #FEF2F2; border-left: 4px solid var(--danger); border-radius: 6px; padding:16px; margin-bottom:12px;">
                <span style="font-weight:700; color:#9B1C1C; text-transform:uppercase; font-size:0.8rem; letter-spacing:0.05em;">Priority 1: Retention Campaigns</span>
                <h5 style="margin: 8px 0 4px 0; color: var(--slate-900);">Address High-Risk Customer Cohorts</h5>
                <p style="margin:0; font-size:0.85rem; color: var(--slate-700); line-height:1.4;">
                    Assign customer success specialists to contact churning customers directly. 
                    Deploy automated re-engagement voucher structures utilizing personalized emails.
                </p>
            </div>
            <div style="background-color: #FFFBEB; border-left: 4px solid var(--warning); border-radius: 6px; padding:16px;">
                <span style="font-weight:700; color:#713F12; text-transform:uppercase; font-size:0.8rem; letter-spacing:0.05em;">Priority 2: VIP Retention</span>
                <h5 style="margin: 8px 0 4px 0; color: var(--slate-900);">Enroll High-Value VIP Segmentations</h5>
                <p style="margin:0; font-size:0.85rem; color: var(--slate-700); line-height:1.4;">
                    Ensure priority shipping logic and early product access privileges are fully mapped 
                    for Cluster 0 profiles to maximize lifetime purchase cycles.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with exec_col2:
        st.markdown("""
            <div style="background-color: #FEE2E2; border-left: 4px solid var(--danger); border-radius: 6px; padding:16px; margin-bottom:12px;">
                <span style="font-weight:700; color:#9B1C1C; text-transform:uppercase; font-size:0.8rem; letter-spacing:0.05em;">Priority 1: Supply Chain Orders</span>
                <h5 style="margin: 8px 0 4px 0; color: var(--slate-900);">Urgent Stockout Replenishment</h5>
                <p style="margin:0; font-size:0.85rem; color: var(--slate-700); line-height:1.4;">
                    Execute orders for critical product codes currently below ROP threshold. 
                    Target expediting freight processes with supplier systems.
                </p>
            </div>
            <div style="background-color: #EFF6FF; border-left: 4px solid var(--primary); border-radius: 6px; padding:16px;">
                <span style="font-weight:700; color:#1E40AF; text-transform:uppercase; font-size:0.8rem; letter-spacing:0.05em;">Priority 3: Forecast Alignment</span>
                <h5 style="margin: 8px 0 4px 0; color: var(--slate-900);">Forecast Horizon Production Scheduling</h5>
                <p style="margin:0; font-size:0.85rem; color: var(--slate-700); line-height:1.4;">
                    Align fulfillment capacity matrices and storage layouts to reflect forecasted 
                    weekly sales targets over the next 8-week horizon.
                </p>
            </div>
        """, unsafe_allow_html=True)
