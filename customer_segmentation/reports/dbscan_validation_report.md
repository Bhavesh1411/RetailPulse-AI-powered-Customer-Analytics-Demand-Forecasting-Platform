# DBSCAN Validation & Anomaly Discovery Report

**Model**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)  
**Purpose**: Validate KMeans findings, discover hidden customer groups, and identify outlier customers  
**Dataset**: Same standardized feature set used for KMeans (4,276 customers, 6 features)  
**KMeans Status**: Remains the primary segmentation model — unchanged

---

## PHASE 1 — DBSCAN PARAMETER SELECTION

### K-Distance Analysis

We performed a k-nearest-neighbor distance analysis to identify the natural density neighborhood radius of the data. For `min_samples=10`, the sorted 10th-nearest-neighbor distances were:

| Percentile | Distance |
| :---: | :---: |
| 50th (Median) | 0.5413 |
| 75th | 0.6816 |
| 80th | 0.7258 |
| 85th | 0.7865 |
| 90th | 0.9627 |
| 92nd | 1.0173 |
| 95th | 1.1338 |
| 97th | 1.2864 |
| 99th | 1.6597 |

> **Interpretation**: The "elbow" in the k-distance curve occurs around the **90th–95th percentile** (distance 0.96–1.13). This means ~90% of customers have their 10th nearest neighbor within a radius of ~1.0, while ~5–10% live in significantly sparser regions — these are the natural outlier candidates.

### Grid Search Results

We tested eps values from 0.4 to 1.5 across min_samples of 7, 10, and 13. Key findings:

- **eps ≥ 1.3**: DBSCAN collapses everything into 1 cluster (no separation). The data forms one continuous density mass at large radii.
- **eps = 0.7–1.0**: DBSCAN finds **2 clusters + noise** — isolating a dense weekend-shopper core and flagging behavioral outliers.
- **eps ≤ 0.5**: DBSCAN fragments into 7+ micro-clusters with 44%+ noise — too aggressive, loses interpretability.

### Selected Parameters

| Parameter | Value | Justification |
| :--- | :---: | :--- |
| **eps** | **0.8** | Positioned at the k-distance elbow. Captures the natural density neighborhood while isolating genuine outliers. |
| **min_samples** | **7** | Set to `n_features + 1` (6+1=7), the standard heuristic for 6-dimensional data. Ensures clusters require meaningful local density. |

---

## PHASE 2 — DBSCAN EXECUTION RESULTS

| Metric | Value |
| :--- | :---: |
| **Clusters Discovered** | **2** |
| **Noise Points** | **541** |
| **Noise Percentage** | **12.7%** |

### Cluster Sizes

| Group | Count | % of Base | Description |
| :--- | :---: | :---: | :--- |
| **DBSCAN Cluster 0** | 3,503 | 81.9% | Main customer mass (mixed VIP + Occasional) |
| **DBSCAN Cluster 1** | 232 | 5.4% | Dense weekend-shopper core |
| **Noise (-1)** | 541 | 12.7% | Behavioral outliers |

### Feature Means by DBSCAN Group (Original Space)

| Group | recency | frequency | monetary | avg_purchase_value | tenure | weekend_ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Noise** | 51.4 | 11.9 | **$6,524** | **$502** | 233 | 0.41 |
| **DBSCAN Cluster 0** | 91.0 | 4.7 | $1,371 | $286 | 232 | 0.06 |
| **DBSCAN Cluster 1** | 138.7 | 1.4 | $331 | $252 | 165 | **1.00** |

> [!IMPORTANT]
> **Key Finding**: The DBSCAN "noise" points have the **highest average monetary value ($6,524)** and the **highest frequency (11.9)** of any group. These are not garbage data — they are the business's most extreme and valuable customers.

---

## PHASE 3 — COMPARISON WITH KMEANS

### Cross-Tabulation: KMeans vs. DBSCAN

| | DBSCAN Noise | DBSCAN Cluster 0 | DBSCAN Cluster 1 | Total |
| :--- | :---: | :---: | :---: | :---: |
| **KMeans Cluster 0** (VIP Champions) | **250** | 1,287 | 0 | 1,537 |
| **KMeans Cluster 1** (Lapsing Buyers) | 77 | 2,039 | 0 | 2,116 |
| **KMeans Cluster 2** (Weekend Shoppers) | 214 | 177 | **232** | 623 |
| **Total** | 541 | 3,503 | 232 | 4,276 |

### Analysis Questions

#### 1. Did DBSCAN discover the same customer behaviors?
**Partially.** DBSCAN successfully identified the Weekend Shoppers as a density-separated cluster (DBSCAN Cluster 1 = 232 customers, 100% from KMeans Cluster 2). However, DBSCAN could **not** separate VIP Champions from Lapsing Buyers — it merged them into one large cluster (DBSCAN Cluster 0). This is because in standardized feature space, VIP and Lapsing customers form a continuous density gradient (there is no density gap between them), and DBSCAN can only separate groups with clear density boundaries.

#### 2. Did DBSCAN discover any new customer groups?
**Yes — DBSCAN Cluster 1 reveals a "Pure Weekend" sub-segment.** While KMeans Cluster 2 had 623 weekend shoppers, DBSCAN's density analysis shows that only **232 of them** form a truly dense, homogeneous core with a weekend ratio of exactly **1.00** (100% weekend purchases). The remaining 391 weekend-leaning customers (214 flagged as noise, 177 absorbed into DBSCAN Cluster 0) are more diverse in their behavior and don't cluster as tightly.

#### 3. Did DBSCAN identify outlier customers?
**Yes — 541 customers (12.7%) were flagged as noise/outliers.** This is DBSCAN's primary contribution. These noise points are customers whose behavioral profiles are statistically unusual — they don't fit neatly into any dense neighborhood.

#### 4. Did DBSCAN reveal hidden premium customers?
**Yes — emphatically.** Of the 541 noise points:
- **250 are KMeans VIP Champions** (46.2%) — these are the most extreme high-spenders and high-frequency buyers that even within the VIP cluster are statistical outliers.
- **62 are Ultra-High Spenders** (monetary > $10,000) with a combined spend of **$2,659,159**.
- **63 are Extreme-Frequency Buyers** (20+ orders), including one customer with **269 orders**.

---

## PHASE 4 — OUTLIER CUSTOMER ANALYSIS

### Noise Point Profile Summary (541 Customers)

| Statistic | recency | frequency | monetary | avg_purchase_value | tenure | weekend_ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean** | 51.4 | 11.9 | $6,524 | $502 | 233 | 0.41 |
| **Median** | 17.0 | 5.0 | $1,289 | $224 | 264 | 0.38 |
| **Std Dev** | 76.3 | 23.9 | $23,134 | $955 | 124 | 0.35 |
| **Min** | 1 | 1 | $2.95 | $2.95 | 5 | 0.00 |
| **Max** | 369 | 269 | $341,777 | $11,881 | 374 | 1.00 |

### Noise Sub-Categories

| Category | Count | Description |
| :--- | :---: | :--- |
| **Ultra-High Spenders** (monetary > $10K) | **62** | Enterprise/wholesale-scale buyers |
| **High-Frequency Buyers** (freq > 20 orders) | **63** | Power users with repeat-order behavior |
| **Premium Basket Buyers** (APV > $1K) | **66** | High-ticket-item purchasers |
| **Low-Value Lapsed** (monetary < $100, recency > 300) | **4** | Negligible anomalies |

> [!NOTE]
> Many noise points belong to multiple categories simultaneously (e.g., a customer with 90 orders and $82K spend is both ultra-high and high-frequency).

### Top 10 Outlier Customers by Revenue

| Rank | Customer ID | Recency | Frequency | Monetary | Avg Purchase Value | KMeans Segment |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 18102 | 1 day | 95 orders | **$341,777** | $3,598 | VIP Champion |
| 2 | 14646 | 10 days | 87 orders | **$243,853** | $2,803 | VIP Champion |
| 3 | 14156 | 7 days | 138 orders | **$183,164** | $1,327 | VIP Champion |
| 4 | 14911 | 1 day | 269 orders | **$137,696** | $512 | VIP Champion |
| 5 | 13694 | 9 days | 105 orders | **$128,172** | $1,221 | VIP Champion |
| 6 | 17511 | 3 days | 42 orders | **$83,761** | $1,994 | VIP Champion |
| 7 | 15061 | 3 days | 90 orders | **$82,163** | $913 | VIP Champion |
| 8 | 16684 | 15 days | 34 orders | **$75,610** | $2,224 | VIP Champion |
| 9 | 13089 | 4 days | 132 orders | **$55,802** | $423 | VIP Champion |
| 10 | 16754 | 8 days | 35 orders | **$54,558** | $1,559 | VIP Champion |

### Ultra-High Spender Revenue Impact

| Metric | Value |
| :--- | :---: |
| **Number of Ultra-High Spenders** | 62 customers |
| **Combined Spend** | **$2,659,159** |
| **Average Spend** | $42,890 per customer |
| **Maximum Single Customer Spend** | $341,777 |
| **Average Order Frequency** | 54.6 orders |
| **% of Customer Base** | 1.4% |
| **Estimated % of Total Revenue** | **~31.6%** |

> [!WARNING]
> **Critical Business Risk**: Just **62 customers (1.4% of the base)** account for an estimated **~31.6% of total revenue**. These are likely enterprise, wholesale, or institutional buyers who should be managed as key accounts, not as part of a mass-market customer segment.

---

## PHASE 5 — ADDITIONAL BUSINESS INSIGHTS

These insights were **not** identified through KMeans and are unique to the DBSCAN density-based analysis.

---

### INSIGHT 1 — The Hidden Enterprise Tier
> **Observation**: DBSCAN identified 62 ultra-high-value customers ($10K+ spend) who were previously lumped into the broad "VIP Loyal Champions" KMeans cluster. Their average spend of $42,890 is **9.4× higher** than the overall VIP Cluster 0 average of $4,581. These customers behave fundamentally differently — they are not "loyal retail shoppers" but rather **enterprise, wholesale, or institutional buyers**.
>
> **Business Impact**: Treating them as regular VIP retail customers means they receive generic marketing and service — not the dedicated account management their revenue contribution demands. Losing even 5 of these customers could mean a $200K+ revenue hit.
>
> **Recommended Action**: Extract these 62 customers into a **dedicated "Enterprise Accounts" tier** with assigned account managers, custom pricing, priority fulfillment, and quarterly business reviews.

---

### INSIGHT 2 — The Power-User Phenomenon (269-Order Customer)
> **Observation**: Customer #14911 placed **269 orders** with $137,696 in total spend, averaging $512 per order — the highest frequency buyer in the entire dataset. Several other customers exceed 100+ orders. These customers are not "frequent shoppers" — they are likely **automated reorder accounts, B2B procurement channels, or reseller operations**.
>
> **Business Impact**: These accounts likely require different fulfillment logistics, inventory planning, and commercial terms. Standard retail processes may be inefficient for servicing 100+ order accounts.
>
> **Recommended Action**: Flag accounts with 50+ orders for a **B2B/reseller review**. Determine whether they should be migrated to a wholesale channel with volume-based pricing and streamlined reorder workflows.

---

### INSIGHT 3 — Weekend Shoppers Are Not Homogeneous
> **Observation**: KMeans identified 623 "Weekend Shoppers." DBSCAN refined this: only **232 customers (37%)** form a tight, pure-weekend core (weekend ratio = 1.00). The remaining **391 customers** are behaviorally more diverse — 214 were flagged as outliers and 177 were absorbed into the main cluster. This suggests the "Weekend Shopper" label applies perfectly to only a core third; the others are weekend-leaning but not exclusively so.
>
> **Business Impact**: A weekend campaign targeting all 623 "Weekend Shoppers" would be well-targeted for 232 core customers but only partially relevant for the remaining 391. This dilutes campaign effectiveness and ROI measurement.
>
> **Recommended Action**: Create a **two-tier weekend strategy**: (1) Pure Weekend promotions for the 232-customer core, and (2) broader "Weekend + Weekday" hybrid campaigns for the remaining 391 weekend-leaning customers.

---

### INSIGHT 4 — The Outlier Revenue Concentration is More Extreme Than KMeans Revealed
> **Observation**: KMeans showed that Cluster 0 (35.9% of customers) generates ~85% of revenue. DBSCAN sharpens this further: within Cluster 0, the **250 outlier VIPs** (16.3% of Cluster 0, just 5.8% of total customers) represent a disproportionate share of that revenue. The top 62 alone account for $2.66M. Revenue concentration is not just at the segment level — it's at the **individual customer level** within the segment.
>
> **Business Impact**: The business's revenue curve follows a steep power-law distribution. Standard "segment-level" strategies miss the magnitude of individual customer importance at the tail.
>
> **Recommended Action**: Implement **individual-level CLV monitoring** for the top 100 customers by revenue. Set automated alerts if any top-100 customer's purchase frequency drops by 20%+ from their rolling average.

---

### INSIGHT 5 — Premium Basket Buyers as an Upsell Signal
> **Observation**: DBSCAN identified **66 customers** with an average purchase value exceeding $1,000 per transaction. These "Premium Basket Buyers" purchase high-ticket items or large quantities per order. Their average order value is 3.2× higher than the overall VIP average of $403, but their frequency varies widely (some order frequently, others infrequently).
>
> **Business Impact**: These customers demonstrate a proven willingness to make large single transactions. Low-frequency premium buyers represent an **under-monetized segment** — they buy big when they buy, but they don't buy often enough.
>
> **Recommended Action**: Target the low-frequency premium buyers (APV > $1K, frequency < 5) with **event-triggered campaigns** (new product launches, exclusive previews, premium collections) to increase their purchase frequency from 2–3 to 4–5 times per year. Even one additional $1K order from 30 of these customers adds $30K+ in revenue.

---

## PHASE 6 — FINAL RECOMMENDATION

### Model Comparison Summary

| Criterion | KMeans (K=3) | DBSCAN (eps=0.8, ms=7) |
| :--- | :--- | :--- |
| **Clusters Found** | 3 clean segments | 2 clusters + 541 noise points |
| **Segment Clarity** | Excellent — 3 distinct personas | Poor — merges VIP and Lapsing into 1 blob |
| **Business Interpretability** | High — each cluster maps to a clear persona | Low for clusters, High for outliers |
| **Outlier Detection** | None — outliers absorbed into clusters | Excellent — 541 outliers with detailed profiles |
| **Stability** | High — deterministic with fixed seed | Moderate — sensitive to eps/min_samples |
| **Actionability** | High — directly maps to marketing campaigns | High for outlier management, low for mass campaigns |
| **Weekend Shoppers** | Identified 623 (broad definition) | Refined to 232 pure-core + 391 leaning |

### Final Recommendation: **Hybrid Approach**

Neither model alone captures the full picture. We recommend a **layered strategy**:

| Use Case | Recommended Model | Rationale |
| :--- | :--- | :--- |
| **Business users & dashboards** | **KMeans** | Clean 3-segment framework is intuitive, reportable, and easy to act on. |
| **Marketing campaigns** | **KMeans** | Segment-level targeting (VIP, Lapsing, Weekend) maps directly to campaign design. |
| **Executive decision-making** | **KMeans + DBSCAN outlier overlay** | Present the 3 segments for strategic planning, but flag the 62 enterprise-tier customers as a separate key-account layer. |
| **Outlier & risk management** | **DBSCAN** | The 541 noise points should be monitored individually for churn risk (enterprise accounts) and anomaly detection. |
| **Key account management** | **DBSCAN outlier list** | The top 62 ultra-high spenders should be extracted into a dedicated "Enterprise Tier" managed outside the mass-market segmentation framework. |
| **Weekend campaign refinement** | **DBSCAN refinement** | Use the 232 pure-weekend core for precision targeting; use the broader 623 KMeans group for wider weekend campaigns. |

### Recommended Segmentation Architecture

```
┌─────────────────────────────────────────────────────┐
│            RETAILPULSE CUSTOMER TIERS                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  TIER 1: Enterprise Accounts (62 customers)         │
│  ├── Source: DBSCAN outlier detection                │
│  ├── Revenue: ~$2.66M (est. ~31.6% of total)        │
│  └── Management: Dedicated account managers          │
│                                                     │
│  TIER 2: VIP Loyal Champions (1,475 customers)       │
│  ├── Source: KMeans Cluster 0 minus Enterprise       │
│  ├── Revenue: ~$4.38M estimated                      │
│  └── Management: Loyalty program + retention CRM     │
│                                                     │
│  TIER 3: Weekend Value Shoppers (623 customers)      │
│  ├── Source: KMeans Cluster 2                        │
│  ├── Core: 232 pure-weekend (DBSCAN validated)       │
│  └── Management: Weekend-timed campaigns             │
│                                                     │
│  TIER 4: Lapsing Occasional Buyers (2,116 customers) │
│  ├── Source: KMeans Cluster 1                        │
│  └── Management: Win-back & lifecycle automation     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

*This report is now complete. Awaiting approval before proceeding to the forecasting and inventory optimization phases.*
