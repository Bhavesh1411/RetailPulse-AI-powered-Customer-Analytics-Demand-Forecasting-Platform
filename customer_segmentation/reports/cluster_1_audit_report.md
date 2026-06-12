# Cluster 1 Anomaly Audit Report

This report presents a deep-dive investigation into **Cluster 1** (118 customers) to determine if it represents a genuine business segment or an anomaly.

## 1. Cluster 1 vs. Overall Dataset Comparison

We compared the 118 customers in Cluster 1 against the overall dataset means:

| Metric | Overall Dataset Mean | Cluster 1 Mean |
| :--- | :---: | :---: |
| **Monetary Value** | \$1,920.35 | **\$2.14** |
| **Average Purchase Value** | \$303.80 | **\$1.37** |
| **Frequency** | 5.38 orders | 2.14 orders |
| **Recency** | 91.37 days | 209.12 days |
| **Customer Tenure** | 228.58 days | 234.26 days |

> [!WARNING]
> **Observation**: The monetary value and average purchase value are astronomically lower than the dataset average, despite having a moderate frequency of 2.14 orders.

## 2. Cluster 1 Deep-Dive Checks

We ran specific boundary checks on the monetary and quantity features for Cluster 1:

- **Minimum Monetary Value**: $0.00
- **Maximum Monetary Value**: $36.15
- **Median Monetary Value**: $0.00
- **Customers with EXACTLY $0.00 Spend**: **103** (out of 118)
- **Minimum Quantity Per Order**: -393.0
- **Customers with Negative Average Quantities**: **81**

## 3. Sample Records from Cluster 1

Below are random sample records from Cluster 1. Notice the zero monetary values paired with negative quantities.

| customer_id | monetary | frequency | recency | avg_purchase_value | quantity_per_order |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 12346 | **0.00** | 6 | 67 | 0.00 | 1.50 |
| 12382 | **0.00** | 1 | 318 | 0.00 | **-1.00** |
| 12467 | **0.00** | 2 | 22 | 0.00 | 0.00 |
| 12590 | **0.00** | 2 | 137 | 0.00 | **-1.00** |
| 12896 | **0.00** | 1 | 366 | 0.00 | **-5.00** |
| 12918 | **0.00** | 3 | 262 | 0.00 | **-0.33** |
| 12846 | 15.58 | 1 | 318 | 15.58 | 1.00 |

---

## 4. Conclusion & Explanation

### Is Cluster 1 a legitimate business segment?
**No. Cluster 1 is a data-quality and preprocessing artifact.**

### Evidence and Root Cause:
1. **Refund-Heavy / Cancelled Orders**: The presence of negative `quantity_per_order` values (-5.0, -1.0) indicates that these customers primarily executed refunds or return transactions. 
2. **Preprocessing Artifacts**: During the data cleaning phase (`clean_customer_features.py`), negative monetary values (which occur when refunds exceed purchases) were clipped using a lower bound of `0.0`. 
   - *Result*: 103 customers ended up with an exact monetary value of `$0.00`.
3. **KMeans Behavior**: Because these 118 customers had nearly identical values across multiple dimensions (`monetary = 0`, `avg_purchase_value = 0`, negative/zero quantities), KMeans perfectly grouped them together as an outlier cluster. Their extreme geometric proximity to zero created a dense micro-cluster separate from legitimate low-value shoppers.

### Next Steps:
Since we are evaluating business profiles, we should treat **Cluster 1** as the **"Refund / Data Anomaly"** group and exclude them from marketing strategies, focusing instead on Clusters 0, 2, and 3 for genuine business profiling.

We await your approval to proceed to the Business Profiling phase.
