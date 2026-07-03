"""
=============================================================================
RetailPulse - Final Inventory Optimization Engine
=============================================================================
Module      : inventory_optimization
Script      : run_inventory_optimization.py

DATE RANGE VERIFIED DIRECTLY FROM SOURCE:
  Earliest transaction date : 2009-01-12
  Latest transaction date   : 2010-12-11
  Total calendar days       : 699

DISCLAIMER: Current stock levels are SIMULATED because actual inventory
balance data was not available in the source dataset. Inventory optimization
outputs should be interpreted as a decision-support simulation rather than
a live inventory audit.

BUSINESS ASSUMPTIONS:
  ABC Classification  | Revenue Contribution
    Class A             Top 80%  | SL=99%  | Z=2.33 | LT=7  days
    Class B             Next 15% | SL=95%  | Z=1.65 | LT=10 days
    Class C             Bottom 5%| SL=90%  | Z=1.28 | LT=14 days

  EOQ Parameters:
    Ordering Cost (S) = $50 per order
    Holding Cost Rate = 20% annually

ISOLATION: This script reads ONLY from processed_data/ and writes ONLY
to inventory_optimization/. No other project files are modified.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import json
from scipy.stats import norm

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IO_DIR     = os.path.join(BASE_DIR, "inventory_optimization")
DATASETS_DIR = os.path.join(IO_DIR, "datasets")
OUTPUTS_DIR  = os.path.join(IO_DIR, "outputs")
REPORTS_DIR  = os.path.join(IO_DIR, "reports")
ARTIFACTS_DIR= os.path.join(IO_DIR, "artifacts")

for d in [DATASETS_DIR, OUTPUTS_DIR, REPORTS_DIR, ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)

SALES_FILE   = os.path.join(BASE_DIR, "processed_data", "cleaned_sales_dataset.csv")
PRODUCT_FILE = os.path.join(BASE_DIR, "processed_data", "product_features.csv")

# ---------------------------------------------------------------------------
# VERIFIED DATE RANGE (confirmed directly from source)
# ---------------------------------------------------------------------------
EARLIEST_DATE        = "2009-01-12"
LATEST_DATE          = "2010-12-11"
TOTAL_CALENDAR_DAYS  = 699          # (2010-12-11 - 2009-01-12).days + 1

# ---------------------------------------------------------------------------
# ABC POLICY PARAMETERS
# ---------------------------------------------------------------------------
ABC_POLICY = {
    "A": {"service_level": 0.99, "z_score": 2.33, "lead_time": 7},
    "B": {"service_level": 0.95, "z_score": 1.65, "lead_time": 10},
    "C": {"service_level": 0.90, "z_score": 1.28, "lead_time": 14},
}

# EOQ PARAMETERS
ORDERING_COST       = 50.0    # $ per order
HOLDING_COST_RATE   = 0.20    # 20% annually
MIN_UNIT_COST       = 0.01    # floor to prevent div-by-zero

# Stock simulation seed (reproducible)
STOCK_SEED          = 42

DISCLAIMER_TEXT = (
    "Current stock levels are simulated because actual inventory balance data "
    "was not available in the source dataset. Inventory optimization outputs "
    "should be interpreted as a decision-support simulation rather than a live "
    "inventory audit."
)

# ---------------------------------------------------------------------------
# STEP 1 – LOAD DATA
# ---------------------------------------------------------------------------
def load_data():
    print("=" * 70)
    print("STEP 1: Loading data...")
    print(f"  Sales file   : {SALES_FILE}")
    print(f"  Product file : {PRODUCT_FILE}")

    sales_df = pd.read_csv(SALES_FILE)
    sales_df['date'] = pd.to_datetime(sales_df['date'], dayfirst=True, format='mixed')

    # Verify date range matches verified constants
    actual_min = sales_df['date'].min().date()
    actual_max = sales_df['date'].max().date()
    actual_days = (sales_df['date'].max() - sales_df['date'].min()).days + 1

    print(f"\n  DATE RANGE VERIFICATION (direct from source):")
    print(f"    Earliest transaction date : {actual_min}")
    print(f"    Latest transaction date   : {actual_max}")
    print(f"    Total calendar days       : {actual_days}")

    if actual_days != TOTAL_CALENDAR_DAYS:
        print(f"  [WARNING] Expected {TOTAL_CALENDAR_DAYS} days but found {actual_days}. "
              f"Using actual value.")
        total_days = actual_days
    else:
        total_days = TOTAL_CALENDAR_DAYS
        print(f"  [OK] Date range matches verified constant ({total_days} days).")

    prod_df = pd.read_csv(PRODUCT_FILE)
    print(f"\n  Products in product_features.csv : {len(prod_df):,}")
    print(f"  Unique products in sales data    : {sales_df['product_id'].nunique():,}")

    return sales_df, prod_df, total_days

# ---------------------------------------------------------------------------
# STEP 2 – COMPUTE PRODUCT DEMAND STATISTICS
# ---------------------------------------------------------------------------
def compute_demand_stats(sales_df, total_days):
    print("\n" + "=" * 70)
    print("STEP 2: Computing demand statistics over verified date range...")

    # Aggregate total revenue and quantity per product
    agg = sales_df.groupby('product_id').agg(
        total_revenue   = ('sales_amount',   'sum'),
        total_quantity  = ('quantity_sold',  'sum'),
    ).reset_index()

    # Average daily demand over the FULL verified calendar window (699 days)
    # Clip to 0: net-return products have negative total_quantity and cannot have
    # a negative demand rate for inventory planning purposes.
    agg['avg_daily_demand'] = np.maximum(0.0, agg['total_quantity'] / total_days)

    # Daily demand std dev: standard deviation of per-day quantities
    # (includes zero-demand days implicitly via the correct variance formula)
    # Variance = (SumOfSquares - (Sum^2 / N)) / (N - 1)
    daily_qty = sales_df.groupby(['product_id', sales_df['date'].dt.date])['quantity_sold'].sum()
    sum_sq = daily_qty.groupby('product_id').apply(lambda x: (x**2).sum()).rename('sum_sq')
    n_obs  = daily_qty.groupby('product_id').count().rename('n_obs')   # trading days only

    agg = agg.merge(sum_sq.reset_index(),  on='product_id', how='left')
    agg = agg.merge(n_obs.reset_index(),   on='product_id', how='left')
    agg['sum_sq'] = agg['sum_sq'].fillna(0.0)
    agg['n_obs']  = agg['n_obs'].fillna(1).astype(int)

    # Use total_days (N=699) as the denominator for population-level std dev
    # This accounts for zero-demand days not present in the transaction log
    S  = agg['total_quantity']
    SS = agg['sum_sq']
    N  = total_days
    var = np.maximum(0.0, (SS - (S ** 2) / N) / (N - 1))
    agg['demand_std_dev'] = np.sqrt(var)

    agg.drop(columns=['sum_sq', 'n_obs'], inplace=True)

    # Unit cost = volume-weighted average unit price
    revenue_qty = sales_df.groupby('product_id').apply(
        lambda x: x['sales_amount'].sum() / max(x['quantity_sold'].sum(), 1)
    ).rename('unit_cost').reset_index()
    agg = agg.merge(revenue_qty, on='product_id', how='left')
    agg['unit_cost'] = agg['unit_cost'].clip(lower=MIN_UNIT_COST)

    print(f"  Products processed : {len(agg):,}")
    print(f"  Avg daily demand   : {agg['avg_daily_demand'].mean():.4f} units/day/product")
    print(f"  Avg demand std dev : {agg['demand_std_dev'].mean():.4f}")

    return agg

# ---------------------------------------------------------------------------
# STEP 3 – ABC CLASSIFICATION
# ---------------------------------------------------------------------------
def abc_classification(df):
    print("\n" + "=" * 70)
    print("STEP 3: ABC Classification by lifetime revenue contribution...")

    df = df.sort_values('total_revenue', ascending=False).copy()
    total_rev = df['total_revenue'].sum()
    df['revenue_pct']     = df['total_revenue'] / total_rev
    df['cum_revenue_pct'] = df['revenue_pct'].cumsum()

    def assign_class(cum_pct):
        if cum_pct <= 0.80:
            return 'A'
        elif cum_pct <= 0.95:
            return 'B'
        else:
            return 'C'

    df['abc_class'] = df['cum_revenue_pct'].apply(assign_class)

    summary = df.groupby('abc_class').agg(
        product_count = ('product_id', 'count'),
        total_revenue = ('total_revenue', 'sum'),
    ).reset_index()
    summary['revenue_pct'] = (summary['total_revenue'] / total_rev * 100).round(2)
    summary['pct_of_catalog'] = (summary['product_count'] / len(df) * 100).round(2)

    print(f"\n  {'Class':<8} {'Products':>10} {'% Catalog':>12} {'Revenue %':>12}")
    print(f"  {'-'*44}")
    for _, row in summary.iterrows():
        print(f"  {row['abc_class']:<8} {row['product_count']:>10,} "
              f"{row['pct_of_catalog']:>11.1f}% {row['revenue_pct']:>11.1f}%")

    return df

# ---------------------------------------------------------------------------
# STEP 4 – SAFETY STOCK, ROP, EOQ
# ---------------------------------------------------------------------------
def compute_inventory_metrics(df):
    print("\n" + "=" * 70)
    print("STEP 4: Computing Safety Stock, Reorder Point, and EOQ...")

    # Map ABC policy parameters
    df['z_score']   = df['abc_class'].map(lambda c: ABC_POLICY[c]['z_score'])
    df['lead_time'] = df['abc_class'].map(lambda c: ABC_POLICY[c]['lead_time'])
    df['service_level'] = df['abc_class'].map(lambda c: ABC_POLICY[c]['service_level'])

    # Safety Stock = Z * sigma_d * sqrt(LT)
    df['safety_stock'] = df['z_score'] * df['demand_std_dev'] * np.sqrt(df['lead_time'])

    # Reorder Point = (avg_daily_demand * LT) + SS
    df['reorder_point'] = (df['avg_daily_demand'] * df['lead_time']) + df['safety_stock']

    # EOQ = sqrt(2 * D_annual * S / H)
    # annual_demand must be >= 0 (net-return products already clipped at demand stage)
    df['annual_demand'] = np.maximum(0.0, df['avg_daily_demand'] * 365.0)
    df['holding_cost_per_unit'] = HOLDING_COST_RATE * df['unit_cost']
    # Guard against zero holding cost
    df['holding_cost_per_unit'] = df['holding_cost_per_unit'].clip(lower=MIN_UNIT_COST * HOLDING_COST_RATE)
    # Guard numerator >= 0 before sqrt to prevent NaN/negative under radical
    numerator = np.maximum(0.0, 2.0 * df['annual_demand'] * ORDERING_COST)
    df['eoq'] = np.sqrt(numerator / df['holding_cost_per_unit'])

    # Validation checks
    assert (df['safety_stock'] >= 0).all(),   "FAIL: Negative safety_stock detected"
    assert (df['reorder_point'] >= 0).all(),  "FAIL: Negative reorder_point detected"
    assert (df['eoq'] >= 0).all(),            "FAIL: Negative eoq detected"
    assert not df['safety_stock'].isna().any(),"FAIL: NaN in safety_stock"
    assert not df['reorder_point'].isna().any(),"FAIL: NaN in reorder_point"
    assert not df['eoq'].isna().any(),        "FAIL: NaN in eoq"

    print(f"  [OK] All validation checks passed.")
    print(f"  Safety Stock  — mean: {df['safety_stock'].mean():.2f}, "
          f"max: {df['safety_stock'].max():.2f}")
    print(f"  Reorder Point — mean: {df['reorder_point'].mean():.2f}, "
          f"max: {df['reorder_point'].max():.2f}")
    print(f"  EOQ           — mean: {df['eoq'].mean():.2f}, "
          f"max: {df['eoq'].max():.2f}")

    return df

# ---------------------------------------------------------------------------
# STEP 5 – SIMULATE CURRENT STOCK
# ---------------------------------------------------------------------------
def simulate_current_stock(df):
    print("\n" + "=" * 70)
    print("STEP 5: Simulating current stock levels (seed=42, reproducible)...")
    print(f"  DISCLAIMER: {DISCLAIMER_TEXT}")

    rng = np.random.default_rng(STOCK_SEED)
    # Multiplier between 0.1 and 2.0 applied to (SS + LT demand)
    multiplier = rng.uniform(0.1, 2.0, size=len(df))
    base = df['safety_stock'] + (df['avg_daily_demand'] * df['lead_time'])
    df['simulated_current_stock'] = np.maximum(0.0, base * multiplier)

    print(f"  Simulated stock — mean: {df['simulated_current_stock'].mean():.2f}, "
          f"max: {df['simulated_current_stock'].max():.2f}")

    return df

# ---------------------------------------------------------------------------
# STEP 6 – STOCKOUT RISK ANALYSIS
# ---------------------------------------------------------------------------
def compute_stockout_risk(df):
    print("\n" + "=" * 70)
    print("STEP 6: Computing Stockout Risk metrics...")

    d  = df['avg_daily_demand']
    lt = df['lead_time']
    ss = df['safety_stock']
    rop= df['reorder_point']
    s  = df['simulated_current_stock']
    sigma_d = df['demand_std_dev']

    # Days to stockout
    df['days_to_stockout'] = np.where(d > 0, s / d, 999.0)

    # Reorder urgency = how far below ROP are we (as ratio)?
    # If stock >= ROP then urgency = 0
    df['reorder_urgency'] = np.where(s < rop, (rop - s) / (rop + 1e-6), 0.0)

    # Stockout probability during lead time window
    # P(demand_during_LT > stock) = 1 - Phi((S - d*LT) / (sigma_d * sqrt(LT)))
    demand_during_lt = d * lt
    std_during_lt    = sigma_d * np.sqrt(lt) + 1e-5
    z_val = (s - demand_during_lt) / std_during_lt
    df['stockout_probability'] = norm.sf(z_val)   # 1 - CDF = sf (survival function)
    df['stockout_probability']  = df['stockout_probability'].clip(0.0, 1.0).round(4)

    # Stockout Risk Category
    def risk_category(row):
        stk = row['simulated_current_stock']
        ss_ = row['safety_stock']
        rop_= row['reorder_point']
        if stk < ss_:
            return 'Critical'
        elif stk < rop_:
            return 'High'
        elif stk < 1.5 * rop_:
            return 'Medium'
        else:
            return 'Low'

    df['risk_category'] = df.apply(risk_category, axis=1)

    risk_counts = df['risk_category'].value_counts()
    print(f"\n  Risk Category Distribution:")
    for cat in ['Critical', 'High', 'Medium', 'Low']:
        cnt = risk_counts.get(cat, 0)
        pct = cnt / len(df) * 100
        print(f"    {cat:<10}: {cnt:>5,}  ({pct:.1f}%)")

    return df

# ---------------------------------------------------------------------------
# STEP 7 – INVENTORY HEALTH SCORE
# ---------------------------------------------------------------------------
def compute_health_score(df):
    print("\n" + "=" * 70)
    print("STEP 7: Computing Inventory Health Score...")

    def pct_at_risk(class_label):
        subset = df[df['abc_class'] == class_label]
        if len(subset) == 0:
            return 0.0
        at_risk = subset['risk_category'].isin(['Critical', 'High']).sum()
        return at_risk / len(subset) * 100

    a_risk = pct_at_risk('A')
    b_risk = pct_at_risk('B')
    c_risk = pct_at_risk('C')

    health_score = 100 - (0.50 * a_risk + 0.30 * b_risk + 0.20 * c_risk)
    health_score = round(max(0.0, min(100.0, health_score)), 2)

    print(f"  Class A at-risk %  : {a_risk:.1f}%")
    print(f"  Class B at-risk %  : {b_risk:.1f}%")
    print(f"  Class C at-risk %  : {c_risk:.1f}%")
    print(f"  Inventory Health Score: {health_score:.2f} / 100")

    return health_score, a_risk, b_risk, c_risk

# ---------------------------------------------------------------------------
# STEP 8 – EXPORT DATASETS
# ---------------------------------------------------------------------------
def export_datasets(df, health_score, a_risk, b_risk, c_risk, total_days):
    print("\n" + "=" * 70)
    print("STEP 8: Exporting dashboard-ready datasets...")

    # ── 8a. INVENTORY MASTER ─────────────────────────────────────────────
    master_cols = [
        'product_id', 'abc_class', 'service_level', 'lead_time',
        'total_revenue', 'total_quantity',
        'avg_daily_demand', 'annual_demand', 'demand_std_dev',
        'unit_cost',
        'z_score', 'safety_stock', 'reorder_point',
        'eoq', 'holding_cost_per_unit',
        'simulated_current_stock',
        'days_to_stockout', 'reorder_urgency', 'stockout_probability',
        'risk_category',
    ]
    master = df[master_cols].copy()
    master = master.round({
        'avg_daily_demand': 4, 'annual_demand': 2, 'demand_std_dev': 4,
        'unit_cost': 4, 'safety_stock': 2, 'reorder_point': 2,
        'eoq': 2, 'holding_cost_per_unit': 4,
        'simulated_current_stock': 2, 'days_to_stockout': 2,
        'reorder_urgency': 4, 'stockout_probability': 4,
    })

    master_path = os.path.join(OUTPUTS_DIR, 'inventory_master.csv')
    master.to_csv(master_path, index=False)
    print(f"  [SAVED] inventory_master.csv  ({len(master):,} rows) → {master_path}")

    # ── 8b. REORDER RECOMMENDATIONS (top 100 by urgency) ─────────────────
    reorder_cols = [
        'product_id', 'abc_class', 'risk_category',
        'simulated_current_stock', 'safety_stock', 'reorder_point',
        'avg_daily_demand', 'lead_time',
        'reorder_urgency', 'days_to_stockout', 'stockout_probability',
        'eoq',
    ]
    reorder = (
        df[df['risk_category'].isin(['Critical', 'High'])]
        [reorder_cols]
        .sort_values(['abc_class', 'reorder_urgency'], ascending=[True, False])
        .head(100)
        .copy()
    )
    reorder = reorder.round({
        'simulated_current_stock': 2, 'safety_stock': 2,
        'reorder_point': 2, 'avg_daily_demand': 4,
        'reorder_urgency': 4, 'days_to_stockout': 2,
        'stockout_probability': 4, 'eoq': 2,
    })
    reorder_path = os.path.join(OUTPUTS_DIR, 'reorder_recommendations.csv')
    reorder.to_csv(reorder_path, index=False)
    print(f"  [SAVED] reorder_recommendations.csv  ({len(reorder):,} rows) → {reorder_path}")

    # ── 8c. STOCKOUT RISK REPORT ──────────────────────────────────────────
    risk_cols = [
        'product_id', 'abc_class', 'risk_category',
        'simulated_current_stock', 'safety_stock', 'reorder_point',
        'avg_daily_demand', 'demand_std_dev', 'lead_time',
        'stockout_probability', 'days_to_stockout',
    ]
    risk_report = df[risk_cols].copy()
    risk_report = risk_report.sort_values('stockout_probability', ascending=False)
    risk_report = risk_report.round({
        'simulated_current_stock': 2, 'safety_stock': 2,
        'reorder_point': 2, 'avg_daily_demand': 4,
        'demand_std_dev': 4, 'stockout_probability': 4,
        'days_to_stockout': 2,
    })
    risk_path = os.path.join(OUTPUTS_DIR, 'stockout_risk_report.csv')
    risk_report.to_csv(risk_path, index=False)
    print(f"  [SAVED] stockout_risk_report.csv  ({len(risk_report):,} rows) → {risk_path}")

    # ── 8d. KPI SUMMARY ───────────────────────────────────────────────────
    critical_count = (df['risk_category'] == 'Critical').sum()
    high_count     = (df['risk_category'] == 'High').sum()
    medium_count   = (df['risk_category'] == 'Medium').sum()
    low_count      = (df['risk_category'] == 'Low').sum()

    total_inventory_value = (df['simulated_current_stock'] * df['unit_cost']).sum()
    excess_stock_value    = (
        np.maximum(0, df['simulated_current_stock'] - df['reorder_point']) * df['unit_cost']
    ).sum()

    abc_counts = df.groupby('abc_class').size().to_dict()

    kpi = {
        "verified_date_range": {
            "earliest_date": EARLIEST_DATE,
            "latest_date":   LATEST_DATE,
            "total_calendar_days": total_days,
        },
        "total_products": len(df),
        "inventory_health_score": health_score,
        "abc_class_distribution": {
            k: {"count": abc_counts.get(k, 0),
                "pct_at_risk": round(
                    {"A": a_risk, "B": b_risk, "C": c_risk}[k], 2
                )}
            for k in ['A', 'B', 'C']
        },
        "risk_category_counts": {
            "Critical": int(critical_count),
            "High":     int(high_count),
            "Medium":   int(medium_count),
            "Low":      int(low_count),
        },
        "total_simulated_inventory_value_usd":   round(total_inventory_value, 2),
        "excess_stock_value_usd":                round(excess_stock_value, 2),
        "avg_days_to_stockout_class_a":          round(df[df['abc_class']=='A']['days_to_stockout'].median(), 2),
        "avg_days_to_stockout_class_b":          round(df[df['abc_class']=='B']['days_to_stockout'].median(), 2),
        "avg_days_to_stockout_class_c":          round(df[df['abc_class']=='C']['days_to_stockout'].median(), 2),
        "avg_eoq_class_a":    round(df[df['abc_class']=='A']['eoq'].mean(), 2),
        "avg_eoq_class_b":    round(df[df['abc_class']=='B']['eoq'].mean(), 2),
        "avg_eoq_class_c":    round(df[df['abc_class']=='C']['eoq'].mean(), 2),
        "disclaimer": DISCLAIMER_TEXT,
    }

    # Save as CSV too
    kpi_rows = []
    for k, v in kpi.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, dict):
                    for ssub_k, ssub_v in sub_v.items():
                        kpi_rows.append({'metric': f"{k}.{sub_k}.{ssub_k}", 'value': ssub_v})
                else:
                    kpi_rows.append({'metric': f"{k}.{sub_k}", 'value': sub_v})
        else:
            kpi_rows.append({'metric': k, 'value': v})
    kpi_df = pd.DataFrame(kpi_rows)

    kpi_path = os.path.join(OUTPUTS_DIR, 'inventory_kpi_summary.csv')
    kpi_df.to_csv(kpi_path, index=False)
    print(f"  [SAVED] inventory_kpi_summary.csv  ({len(kpi_df):,} rows) → {kpi_path}")

    # Also save JSON for downstream use
    kpi_json_path = os.path.join(REPORTS_DIR, 'inventory_kpi_summary.json')
    with open(kpi_json_path, 'w') as f:
        json.dump(kpi, f, indent=4)
    print(f"  [SAVED] inventory_kpi_summary.json → {kpi_json_path}")

    return kpi, master, reorder, risk_report

# ---------------------------------------------------------------------------
# STEP 9 – FINAL MARKDOWN REPORT
# ---------------------------------------------------------------------------
def generate_final_report(df, kpi, master, reorder, risk_report):
    print("\n" + "=" * 70)
    print("STEP 9: Generating Final Inventory Optimization Report...")

    health  = kpi['inventory_health_score']
    abc_d   = kpi['abc_class_distribution']
    risks   = kpi['risk_category_counts']
    total_p = kpi['total_products']

    # Top 10 reorder recommendations
    top10 = reorder.head(10)

    # Format helpers
    def pct(n, total):
        return f"{n/total*100:.1f}%" if total > 0 else "0.0%"

    health_label = (
        "🟢 Excellent" if health >= 85 else
        "🟡 Good"      if health >= 70 else
        "🟠 Fair"      if health >= 55 else
        "🔴 Poor"
    )

    report = f"""# RetailPulse Inventory Optimization — Final Report

> [!IMPORTANT]
> **DISCLAIMER:** {DISCLAIMER_TEXT}

---

## Executive Summary

| Metric | Value |
|:---|---:|
| **Inventory Health Score** | **{health:.2f} / 100** ({health_label}) |
| **Total Products Analyzed** | {total_p:,} |
| **Verified Date Range** | {EARLIEST_DATE} → {LATEST_DATE} ({kpi['verified_date_range']['total_calendar_days']} calendar days) |
| **Simulated Inventory Value** | ${kpi['total_simulated_inventory_value_usd']:,.2f} |
| **Excess Stock Value** | ${kpi['excess_stock_value_usd']:,.2f} |

---

## 1. Date Range Verification

All calculations in this module are based on the **verified transactional date range** read directly from the source file `cleaned_sales_dataset.csv`:

| Parameter | Value |
|:---|:---|
| Earliest Transaction Date | `{EARLIEST_DATE}` |
| Latest Transaction Date | `{LATEST_DATE}` |
| Total Calendar Days | **{kpi['verified_date_range']['total_calendar_days']}** |

> [!NOTE]
> Average daily demand ($d$) and demand standard deviation ($\\sigma_d$) for every product are computed over the full **699-day** calendar window, including zero-demand days not present in the transaction log. This ensures conservative, accurate safety stock estimates.

---

## 2. Business Assumptions

| Class | Service Level | Z-Score | Lead Time (days) |
|:---:|:---:|:---:|:---:|
| **A** | 99% | 2.33 | 7 |
| **B** | 95% | 1.65 | 10 |
| **C** | 90% | 1.28 | 14 |

**EOQ Parameters:**
- Ordering Cost ($S$): **$50.00 per order**
- Holding Cost Rate: **20% annually**
- Unit Cost: Estimated as volume-weighted average selling price (clipped to a minimum of $0.01)

---

## 3. ABC Classification

ABC classification is based on lifetime revenue contribution over the verified date range.

| Class | Products | % of Catalog | Revenue % | Lead Time | Service Level |
|:---:|---:|---:|---:|---:|---:|
| **A** | {abc_d['A']['count']:,} | {abc_d['A']['count']/total_p*100:.1f}% | ~80% | 7 days | 99% |
| **B** | {abc_d['B']['count']:,} | {abc_d['B']['count']/total_p*100:.1f}% | ~15% | 10 days | 95% |
| **C** | {abc_d['C']['count']:,} | {abc_d['C']['count']/total_p*100:.1f}% | ~5% | 14 days | 90% |

> [!TIP]
> The catalog follows the **Pareto Principle** (80/20 rule). Only {abc_d['A']['count']:,} Class A products (~{abc_d['A']['count']/total_p*100:.0f}% of catalog) are responsible for ~80% of total revenue. Tight inventory management of Class A products will protect the vast majority of business value.

---

## 4. Safety Stock Analysis

**Formula:** $SS = Z \\times \\sigma_d \\times \\sqrt{{LT}}$

Safety Stock ensures products remain available during demand spikes while awaiting supplier replenishment.

| Class | Z-Score | Lead Time | Avg $\\sigma_d$ | Avg Safety Stock |
|:---:|:---:|:---:|---:|---:|
| **A** | 2.33 | 7 days | {df[df['abc_class']=='A']['demand_std_dev'].mean():.2f} units | {df[df['abc_class']=='A']['safety_stock'].mean():.2f} units |
| **B** | 1.65 | 10 days | {df[df['abc_class']=='B']['demand_std_dev'].mean():.2f} units | {df[df['abc_class']=='B']['safety_stock'].mean():.2f} units |
| **C** | 1.28 | 14 days | {df[df['abc_class']=='C']['demand_std_dev'].mean():.2f} units | {df[df['abc_class']=='C']['safety_stock'].mean():.2f} units |

---

## 5. Reorder Point Analysis

**Formula:** $ROP = (d \\times LT) + SS$

The Reorder Point triggers a replenishment order before stock is depleted.

| Class | Avg Daily Demand | Lead Time | Avg Safety Stock | Avg ROP |
|:---:|---:|:---:|---:|---:|
| **A** | {df[df['abc_class']=='A']['avg_daily_demand'].mean():.3f} units | 7 days | {df[df['abc_class']=='A']['safety_stock'].mean():.2f} | {df[df['abc_class']=='A']['reorder_point'].mean():.2f} units |
| **B** | {df[df['abc_class']=='B']['avg_daily_demand'].mean():.3f} units | 10 days | {df[df['abc_class']=='B']['safety_stock'].mean():.2f} | {df[df['abc_class']=='B']['reorder_point'].mean():.2f} units |
| **C** | {df[df['abc_class']=='C']['avg_daily_demand'].mean():.3f} units | 14 days | {df[df['abc_class']=='C']['safety_stock'].mean():.2f} | {df[df['abc_class']=='C']['reorder_point'].mean():.2f} units |

---

## 6. Economic Order Quantity (EOQ) Analysis

**Formula:** $EOQ = \\sqrt{{\\frac{{2 \\times D_a \\times S}}{{H}}}}$

where $D_a$ = Annual Demand, $S$ = \\$50 Ordering Cost, $H$ = 20% × Unit Cost.

| Class | Avg Annual Demand | Avg Unit Cost | Avg EOQ |
|:---:|---:|---:|---:|
| **A** | {df[df['abc_class']=='A']['annual_demand'].mean():.1f} units | ${df[df['abc_class']=='A']['unit_cost'].mean():.2f} | {df[df['abc_class']=='A']['eoq'].mean():.1f} units |
| **B** | {df[df['abc_class']=='B']['annual_demand'].mean():.1f} units | ${df[df['abc_class']=='B']['unit_cost'].mean():.2f} | {df[df['abc_class']=='B']['eoq'].mean():.1f} units |
| **C** | {df[df['abc_class']=='C']['annual_demand'].mean():.1f} units | ${df[df['abc_class']=='C']['unit_cost'].mean():.2f} | {df[df['abc_class']=='C']['eoq'].mean():.1f} units |

---

## 7. Stockout Risk Analysis

> [!CAUTION]
> **{risks['Critical']:,} Critical** and **{risks['High']:,} High** risk products require immediate attention. Combined, they represent {pct(risks['Critical']+risks['High'], total_p)} of the catalog.

| Risk Category | Count | % of Catalog | Description |
|:---|---:|---:|:---|
| 🔴 **Critical** | {risks['Critical']:,} | {pct(risks['Critical'], total_p)} | Stock < Safety Stock. Stockout imminent. |
| 🟠 **High** | {risks['High']:,} | {pct(risks['High'], total_p)} | Stock < ROP. Reorder required now. |
| 🟡 **Medium** | {risks['Medium']:,} | {pct(risks['Medium'], total_p)} | Stock ≥ ROP. Approaching reorder threshold. |
| 🟢 **Low** | {risks['Low']:,} | {pct(risks['Low'], total_p)} | Well-stocked. No immediate action. |

### Class A Risk Breakdown
| Risk Category | Count | % of Class A |
|:---|---:|---:|
| 🔴 Critical | {df[(df['abc_class']=='A') & (df['risk_category']=='Critical')].shape[0]:,} | {pct(df[(df['abc_class']=='A') & (df['risk_category']=='Critical')].shape[0], abc_d['A']['count'])} |
| 🟠 High | {df[(df['abc_class']=='A') & (df['risk_category']=='High')].shape[0]:,} | {pct(df[(df['abc_class']=='A') & (df['risk_category']=='High')].shape[0], abc_d['A']['count'])} |
| 🟡 Medium | {df[(df['abc_class']=='A') & (df['risk_category']=='Medium')].shape[0]:,} | {pct(df[(df['abc_class']=='A') & (df['risk_category']=='Medium')].shape[0], abc_d['A']['count'])} |
| 🟢 Low | {df[(df['abc_class']=='A') & (df['risk_category']=='Low')].shape[0]:,} | {pct(df[(df['abc_class']=='A') & (df['risk_category']=='Low')].shape[0], abc_d['A']['count'])} |

---

## 8. Top 10 Immediate Reorder Recommendations

These products are flagged **Critical** or **High** risk and require immediate purchase orders, sorted by reorder urgency within Class A first.

| # | Product ID | Class | Risk | Simulated Stock | ROP | EOQ | Days to Stockout | Stockout Prob |
|:---:|:---|:---:|:---|---:|---:|---:|---:|---:|
"""

    for i, (_, row) in enumerate(top10.iterrows(), start=1):
        report += (
            f"| {i} | {row['product_id']} | {row['abc_class']} | {row['risk_category']} "
            f"| {row['simulated_current_stock']:.2f} | {row['reorder_point']:.2f} "
            f"| {row['eoq']:.2f} | {row['days_to_stockout']:.1f} "
            f"| {row['stockout_probability']:.3f} |\n"
        )

    report += f"""
---

## 9. Inventory Health Score

**Score: {health:.2f} / 100 — {health_label}**

The Inventory Health Score is a weighted composite measuring the proportion of products that are **NOT** in Critical or High-risk status, weighted by ABC class importance:

$$Health\\ Score = 100 - (0.50 \\times \\%A_{{at\\ risk}} + 0.30 \\times \\%B_{{at\\ risk}} + 0.20 \\times \\%C_{{at\\ risk}})$$

| Input | Value |
|:---|---:|
| Class A at-risk % | {kpi['abc_class_distribution']['A']['pct_at_risk']:.1f}% |
| Class B at-risk % | {kpi['abc_class_distribution']['B']['pct_at_risk']:.1f}% |
| Class C at-risk % | {kpi['abc_class_distribution']['C']['pct_at_risk']:.1f}% |
| **Final Health Score** | **{health:.2f} / 100** |

---

## 10. Median Days-to-Stockout by Class

| Class | Median Days to Stockout |
|:---:|---:|
| **A** | {kpi['avg_days_to_stockout_class_a']:.1f} days |
| **B** | {kpi['avg_days_to_stockout_class_b']:.1f} days |
| **C** | {kpi['avg_days_to_stockout_class_c']:.1f} days |

---

## 11. Business Recommendations

1. **Immediate Action — Class A Critical Products:** Order a quantity equal to at least the computed EOQ for every Class A Critical product today. These products drive ~80% of revenue and are at imminent risk.

2. **Trigger-Based Ordering System:** Implement ROP-triggered purchase orders for all Class A and B products. When `Simulated Current Stock` falls below `Reorder Point`, a purchase order for `EOQ` units should be automatically raised.

3. **Excess Stock Reduction:** The estimated simulated excess stock value is **${kpi['excess_stock_value_usd']:,.2f}**. Review Class C products with Low risk and very high stock, and consider reducing order frequencies or offering promotions to clear slow-moving inventory.

4. **Lead Time Negotiation:** Class C products carry the longest lead time (14 days) and the lowest service level (90%). If supplier negotiations can reduce Class C lead time to 10 days, Safety Stock requirements will decrease by approximately 16%, freeing up working capital.

5. **Data Infrastructure Recommendation:** The most critical upgrade for production-grade inventory management is implementing **real-time inventory tracking** (e.g., a warehouse management system). This will replace the simulated current stock with live balances, converting this decision-support simulation into a live operational dashboard.

6. **Integrate Demand Forecasting:** Connect the weekly XGBoost forecast model (Val MAPE: 11.8%) from the Demand Forecasting module. Substituting forecasted demand for historical average daily demand in the ROP calculation will make reorder triggers forward-looking rather than backward-looking.

---

## 12. Output Files

| File | Location | Description |
|:---|:---|:---|
| `inventory_master.csv` | `outputs/` | Full product-level master inventory dataset |
| `reorder_recommendations.csv` | `outputs/` | Top 100 immediate reorder products |
| `stockout_risk_report.csv` | `outputs/` | Risk categorization for all {total_p:,} products |
| `inventory_kpi_summary.csv` | `outputs/` | Macro KPI summary table |
| `inventory_kpi_summary.json` | `reports/` | Machine-readable KPI JSON |

---

## 13. Validation Results

All pre-export validation checks **passed**:

- ✅ No negative Safety Stock values
- ✅ No negative Reorder Point values
- ✅ No negative EOQ values
- ✅ No divide-by-zero errors
- ✅ Product count = **{total_p:,}** (matches source dataset exactly)
- ✅ Verified date range used throughout: {EARLIEST_DATE} → {LATEST_DATE} ({kpi['verified_date_range']['total_calendar_days']} days)

---

## 14. Simulation Disclaimer

> [!WARNING]
> {DISCLAIMER_TEXT}

This module is a **decision-support simulation**. All outputs are based on historical transactional data from `cleaned_sales_dataset.csv`. Simulated current stock levels were generated using a reproducible random seed (seed=42) and should be replaced with live inventory counts before any operational deployment.

---

*Report generated by `run_inventory_optimization.py` — RetailPulse Inventory Optimization Module*
"""

    report_path = os.path.join(REPORTS_DIR, 'final_inventory_optimization_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [SAVED] final_inventory_optimization_report.md → {report_path}")

    return report_path

# ---------------------------------------------------------------------------
# STEP 10 – FINAL VALIDATION CHECKS
# ---------------------------------------------------------------------------
def final_validation(df, expected_count):
    print("\n" + "=" * 70)
    print("STEP 10: Running final validation checks...")

    assert len(df) == expected_count, \
        f"FAIL: Product count mismatch. Expected {expected_count}, got {len(df)}"
    assert (df['safety_stock']  >= 0).all(), "FAIL: Negative safety_stock"
    assert (df['reorder_point'] >= 0).all(), "FAIL: Negative reorder_point"
    assert (df['eoq']           >= 0).all(), "FAIL: Negative eoq"
    assert not df['safety_stock'].isna().any(),  "FAIL: NaN safety_stock"
    assert not df['reorder_point'].isna().any(), "FAIL: NaN reorder_point"
    assert not df['eoq'].isna().any(),           "FAIL: NaN eoq"
    assert not df['simulated_current_stock'].isna().any(), "FAIL: NaN simulated_current_stock"

    print(f"  ✅ No negative Safety Stock values")
    print(f"  ✅ No negative Reorder Point values")
    print(f"  ✅ No negative EOQ values")
    print(f"  ✅ No divide-by-zero errors detected")
    print(f"  ✅ Product count = {len(df):,} (matches source exactly)")
    print(f"  ✅ All validation checks PASSED")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  RetailPulse — Final Inventory Optimization Engine")
    print("=" * 70)
    print(f"  Verified Date Range : {EARLIEST_DATE} to {LATEST_DATE}")
    print(f"  Total Calendar Days : {TOTAL_CALENDAR_DAYS}")
    print(f"  Disclaimer          : Simulated current stock levels")
    print("=" * 70)

    # Load
    sales_df, prod_df, total_days = load_data()
    expected_product_count = sales_df['product_id'].nunique()

    # Build product-level stats
    df = compute_demand_stats(sales_df, total_days)

    # ABC classification
    df = abc_classification(df)

    # Safety Stock, ROP, EOQ
    df = compute_inventory_metrics(df)

    # Simulated current stock
    df = simulate_current_stock(df)

    # Stockout risk
    df = compute_stockout_risk(df)

    # Health score
    health_score, a_risk, b_risk, c_risk = compute_health_score(df)

    # Final validation BEFORE export
    final_validation(df, expected_product_count)

    # Export datasets
    kpi, master, reorder, risk_report = export_datasets(
        df, health_score, a_risk, b_risk, c_risk, total_days
    )

    # Generate report
    report_path = generate_final_report(df, kpi, master, reorder, risk_report)

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETE — Inventory Optimization Module Finished")
    print(f"  Final Report: {report_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
