# KMeans Customer Segmentation Evaluation Report

This report presents the results of the KMeans customer segmentation analysis on the RetailPulse dataset, comparing the model performance with and without the optional feature `active_months`.

---

## 1. Executive Summary & Model Selection

After testing both datasets for cluster counts ($K$) ranging from 2 to 8, we determined that **$K=4$** is the optimal number of segments. 

Furthermore, we recommend the model **Without `active_months`** for the final segmentation. 

### Justification:
1. **Redundancy and Collinearity**: `active_months` is highly correlated with `frequency` ($r=0.69$) and `customer_tenure` ($r=0.54$). Including it does not introduce a new behavioral dimension but rather shifts cluster boundaries slightly.
2. **Model Parsimoniousness**: Excluding `active_months` results in a simpler, cleaner feature set of 6 variables (`recency`, `frequency`, `monetary`, `average_purchase_value`, `customer_tenure`, `weekend_sales_ratio`), which reduces overfitting.
3. **Statistical Comparability**: The silhouette score at $K=4$ is highly comparable between the two models (0.3391 with vs. 0.3250 without), and they yield almost identical cluster counts and unscaled feature averages.

---

## 2. Evaluation Metrics Comparison

| K Value | WCSS (With active_months) | Silhouette Score (With) | DB Index (With) | WCSS (Without active_months) | Silhouette Score (Without) | DB Index (Without) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 20603.09 | 0.3027 | 1.2550 | 19201.88 | 0.2594 | 1.4397 |
| **3** | 16788.65 | 0.3272 | 1.0034 | 15426.69 | 0.2860 | 1.1158 |
| **4** | 13616.49 | **0.3391** | 1.0227 | 12182.11 | **0.3250** | 1.0246 |
| **5** | 11265.54 | 0.3080 | 1.0182 | 9798.70 | 0.3083 | **0.9755** |
| **6** | 9639.97 | 0.2848 | 1.0556 | 8619.66 | 0.2798 | 1.0799 |
| **7** | 8929.98 | 0.2878 | 1.1519 | 7999.84 | 0.2802 | 1.1348 |
| **8** | 8282.81 | 0.2819 | 1.1362 | 7469.94 | 0.2702 | 1.1567 |

### Optimal K Determination:
- **Elbow Curve (WCSS)**: A visible slope reduction (elbow) begins to form at $K=3$ and stabilizes at $K=4$. 
- **Silhouette Coefficient**: Peaks at **$K=4$** for both feature sets (0.3391 and 0.3250).
- **Davies-Bouldin Index**: Approaches its minimum around $K=4$ and $K=5$, signaling good cluster separation and compactness.

---

## 3. Final Selected Model Statistics ($K=4$, Without `active_months`)

The final model was run on the 6 approved standardized features. The resulting customer segment assignments have been saved to:
`C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customer_segments_kmeans.csv`

### Cluster Counts
- **Cluster 0**: 1,472 customers (33.6% of user base)
- **Cluster 1**: 118 customers (2.7% of user base)
- **Cluster 2**: 2,158 customers (49.3% of user base)
- **Cluster 3**: 631 customers (14.4% of user base)

---

## 4. Cluster Profile Data

### A. Scaled Centroids (Scaled Space)
Centroid coordinates in standardized space. These indicate how many standard deviations the cluster mean lies from the overall dataset mean.

| Cluster | recency | frequency | monetary | average_purchase_value | customer_tenure | weekend_sales_ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | -0.709 | 1.083 | 0.851 | 0.280 | 0.718 | -0.086 |
| **Cluster 1** | 0.895 | -0.646 | -3.794 | -4.571 | 0.048 | -0.550 |
| **Cluster 2** | 0.378 | -0.617 | -0.336 | 0.048 | -0.441 | -0.529 |
| **Cluster 3** | 0.196 | -0.295 | -0.127 | 0.038 | -0.178 | 2.113 |

### B. Feature Means by Cluster (Original Space)
The actual average values of the raw features for each cluster.

| Cluster | Count | recency (days) | frequency (orders) | monetary ($) | avg_purchase_value ($) | customer_tenure (days) | weekend_sales_ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | 1,472 | 32.98 | 11.70 | $4,610.84 | $351.87 | 314.43 | 0.112 |
| **Cluster 1** | 118 | 209.12 | 2.14 | $2.14 | $1.37 | 234.26 | 0.000 |
| **Cluster 2** | 2,158 | 122.25 | 1.95 | $536.23 | $298.35 | 175.91 | 0.005 |
| **Cluster 3** | 631 | 99.95 | 2.95 | $736.31 | $266.86 | 207.35 | 0.762 |

---

## 5. Summary of Key Segment Characteristics

* **Cluster 0**: Extremely high purchase frequency (11.7 orders) and overall spend ($4,610.84), lowest recency (32.98 days), and very long tenure (314.4 days). They purchase mostly on weekdays.
* **Cluster 1**: Very small group with high recency (209.1 days) and extremely low total spend ($2.14) and average order value ($1.37). They have zero weekend purchase ratio.
* **Cluster 2**: The largest cluster representing almost half the customer base. They have moderate recency (122.25 days), low order frequency (1.95 orders), and moderate average order values ($298.35). They shop exclusively on weekdays.
* **Cluster 3**: A distinct segment that has a high weekend sales ratio (0.762). They have moderate recency (99.95 days), moderate tenure (207.35 days), and spend around $736.31 on average.

---

## Next Steps

1. **Awaiting User Review**: We will wait for your approval of these cluster assignments and evaluation results.
2. **Business Interpretation**: Once approved, we can assign business segment names (e.g. VIP/Loyalists, Lost Low-Value, Occasional Weekday, Weekend Shoppers) and map out targeted marketing strategies for each segment.
3. **DBSCAN Modeling**: Compare these results to DBSCAN clustering if requested.
