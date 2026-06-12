# Feature Audit and Correlation Analysis

This document provides a comprehensive audit of the available features in `customers_features_finalone.csv` and presents the correlation analysis to recommend the optimal feature set for customer segmentation (clustering).

---

## 1. Feature Audit Table

We have audited all 17 features in the dataset. They are categorized based on their utility for clustering algorithms (e.g., KMeans, DBSCAN), statistical properties, and business relevance.

| Feature Name | DataType | Classification | Description & Statistical Profile | Business Relevance & Decision |
| :--- | :---: | :---: | :--- | :--- |
| **customer_id** | String | **Exclude** | Unique identifier; shape (4379, 17); no missing values. | Excluded as it contains no predictive or behavioral information. Used strictly for mapping clusters back to customers. |
| **recency** | Int64 | **Useful** | Days since last purchase. Mean: ~91 days, Max: 374 days. | **Core RFM Metric**. Essential for identifying active vs. lapsed/churning customers. |
| **frequency** | Int64 | **Useful** | Total number of orders. Mean: ~5.4, Max: 269. | **Core RFM Metric**. Crucial for measuring customer loyalty, engagement, and transactional volume. |
| **monetary** | Float64 | **Useful** | Total spend. Mean: \$1,920, Max: \$341,776. | **Core RFM Metric**. Crucial for identifying financial value and high-yield customer segments. |
| **average_purchase_value** | Float64 | **Useful** | Average basket value per transaction. Mean: \$303.8, Max: \$11,880. | Helps distinguish high-value ticket buyers (fewer but larger orders) from low-value frequent shoppers. |
| **quantity_per_order** | Float64 | **Optional** | Average items per order. Mean: 210, Max: 87,167. | High correlation with average purchase value (0.627). Useful for wholesale vs. retail behavior segmentation. |
| **active_months** | Int64 | **Useful** | Number of months with transactions. Mean: 3.18, Max: 13. | Measures engagement consistency and relationship longevity over time. |
| **purchase_frequency** | Float64 | **Optional** | Average purchases per active month. Mean: 1.42, Max: 20.69. | Highly redundant with `frequency` (0.798). Useful if frequency needs normalization over active span. |
| **customer_loyalty_score** | Float64 | **Exclude** | Composite loyalty score. Mean: 5.51, Max: 10.0. | Engineered metric. High correlation with active months (0.808) and revenue rank (-0.935). Will distort clustering distance metrics. |
| **customer_risk_score** | Float64 | **Exclude** | Churn risk probability score. Mean: 0.32, Max: 1.0. | Engineered probability. High correlation with recency (0.611) and churn warning flag (0.669). |
| **high_value_customer_flag** | Int64 | **Exclude** | Binary indicator for high value. Mean: 0.10 (10% of customers). | Binary flag. Redundant with monetary and active months. Binary flags skew distance metrics in distance-based clustering. |
| **customer_tenure** | Int64 | **Useful** | Total relationship duration in days. Mean: 228.5 days. | Helps segment long-standing loyalists from newly acquired shoppers. |
| **avg_days_between_purchases** | Float64 | **Optional** | Historical order interval. Mean: 112.6 days. | Strong correlation with recency (0.791). Recency represents the current state; avg days represents historical cadence. |
| **weekend_sales_ratio** | Float64 | **Useful** | Ratio of orders placed on weekends. Mean: 0.15, Max: 1.0. | Captures unique customer lifestyle preference (weekday vs. weekend shoppers) with extremely low correlation to other metrics. |
| **customer_rank_by_revenue** | Int64 | **Exclude** | Ordinal revenue rank. Mean: 2189.9. | Redundant ordinal representation of `monetary`. Skews distances drastically due to uniform distribution. |
| **churn_warning_flag** | Int64 | **Exclude** | Binary flag indicating churn risk. Mean: 0.065. | Binary flag. Highly correlated with risk score (0.669) and recency. Exclude to prevent distance distortion. |
| **lost_one_time_buyer_flag** | Int64 | **Exclude** | Binary flag for lapsed one-time purchasers. Mean: 0.166. | Binary flag. High correlation with recency (0.675). Redundant with recency and frequency. |

---

## 2. Correlation Findings

A correlation analysis was performed on all numeric features. The matrix identifies several strong relationships (threshold $|r| \ge 0.5$) that highlight collinearity and redundancies:

### Extreme Redundancy ($|r| \ge 0.8$)
* **`customer_rank_by_revenue` vs. `customer_loyalty_score` ($r = -0.9351$)**:
  * *Interpretation*: A customer's rank (lower is better) is almost perfectly determined by their calculated loyalty score.
  * *Clustering Impact*: Keeping both features would heavily double-weight the exact same underlying concept of customer ranking/loyalty.
* **`customer_loyalty_score` vs. `active_months` ($r = 0.8088$)**:
  * *Interpretation*: Loyalty scores are heavily derived from the number of active months a customer has.

### Strong Correlation ($0.6 \le |r| < 0.8$)
* **`purchase_frequency` vs. `frequency` ($r = 0.7979$)**:
  * *Interpretation*: Frequency and monthly purchase rate are highly aligned. Since frequency represents absolute volume and purchase frequency is rate-based, keeping both is redundant unless normalized.
* **`avg_days_between_purchases` vs. `recency` ($r = 0.7916$)**:
  * *Interpretation*: Customers with long average gaps between purchases naturally exhibit high recency (more days since last purchase).
* **`active_months` vs. `frequency` ($r = 0.6894$)**:
  * *Interpretation*: Customers who purchase more often are naturally active across more months.
* **`lost_one_time_buyer_flag` vs. `recency` ($r = 0.6754$)**:
  * *Interpretation*: High recency (long time since last purchase) is the primary driver of the lost one-time buyer label.
* **`churn_warning_flag` vs. `customer_risk_score` ($r = 0.6695$)**:
  * *Interpretation*: The binary warning flag is directly derived from the continuous risk score.
* **`monetary` vs. `frequency` ($r = 0.6356$)**:
  * *Interpretation*: Customers who order more frequently spend more money overall. This is a standard RFM relationship, but because they capture different dimensions of value (volume vs. cash), they are usually both retained but scaled (e.g., log-transformed).
* **`quantity_per_order` vs. `average_purchase_value` ($r = 0.6270$)**:
  * *Interpretation*: Higher item counts per order lead directly to higher average transaction values.
* **`high_value_customer_flag` vs. `active_months` ($r = 0.6251$)**:
  * *Interpretation*: High-value status is strongly tied to long-term monthly activity.
* **`customer_risk_score` vs. `recency` ($r = 0.6118$)**:
  * *Interpretation*: High risk scores are strongly driven by high recency.

---

## 3. Recommended Feature Set for Clustering

To build the strongest customer segmentation dataset, we recommend using a combination of **Core RFM metrics** and **behavioral preference indicators**, while eliminating redundant, ordinal, and binary flags.

### Propose Core Clustering Features:
1. **`recency`** (Continuous): Measures transactional decay/activeness.
2. **`frequency`** (Continuous): Measures engagement volume.
3. **`monetary`** (Continuous): Measures monetary scale.
4. **`average_purchase_value`** (Continuous): Distinguishes high-value transaction sizes from high-frequency small transactions.
5. **`customer_tenure`** (Continuous): Represents relationship duration, allowing segmentation of long-term vs. newly acquired customers.
6. **`weekend_sales_ratio`** (Continuous/Ratio): Captures buying patterns and lifestyle preferences (weekend vs. weekday).

### Optional / Additional Candidate Features:
* **`active_months`**: Could be included alongside tenure to calculate consistency, but since it is strongly correlated with frequency ($r=0.69$) and loyalty, we must monitor if it dilutes the RFM metrics. We recommend starting without it, as tenure and frequency cover its dimensions.
* **`quantity_per_order`**: Good for separating wholesale/bulk buyers. However, since it correlates heavily with `average_purchase_value` ($r=0.63$), we recommend using `average_purchase_value` as it represents monetary scale better.

---

## 4. Features to Exclude with Reason

We recommend excluding the following features from the clustering input dataset:

| Feature to Exclude | Reason for Exclusion |
| :--- | :--- |
| **customer_id** | **Identifier**: Non-numeric string with no predictive value. Retained in metadata only. |
| **customer_rank_by_revenue** | **Ordinal Rank / Redundant**: Highly correlated with monetary value. Rank values have a flat uniform distribution which distorts Euclidean distance calculations in clustering. |
| **customer_loyalty_score** | **Engineered Metric**: A composite score built from other base features (active months, tenure, frequency). Including it double-counts those features and dominates the distance matrix. |
| **customer_risk_score** | **Engineered Metric**: Churn probability derived from recency. Let the clustering model discover high-risk groups naturally through raw recency and tenure, rather than forcing a pre-computed model's bias. |
| **high_value_customer_flag** | **Binary Target**: Distance-based algorithms (like KMeans) do not handle binary variables well alongside continuous ones. Also redundant with raw monetary/frequency values. |
| **churn_warning_flag** | **Binary Target**: Redundant with recency and customer_risk_score. Binary features distort distance metrics. |
| **lost_one_time_buyer_flag** | **Binary Target**: Redundant with frequency = 1 and high recency. Binary features distort distance metrics. |
| **avg_days_between_purchases** | **High Collinearity**: Extremely correlated with recency ($r = 0.79$). Adding it creates multi-collinearity and over-emphasizes purchase frequency metrics. |
| **purchase_frequency** | **High Collinearity**: Highly correlated with overall frequency ($r = 0.80$). General frequency and active months already describe this behavior. |

---

## Next Steps

1. **Wait for Approval**: We will wait for your review and approval on this recommended feature selection.
2. **Feature Engineering / Transformation**: Once approved, we will prepare the dataset by:
   - Selecting the approved features.
   - Performing Log-Transformation (specifically on highly skewed features like `monetary` and `frequency`).
   - Applying Standard Scaling (Z-score normalization) to ensure equal feature weighting in distance calculations.
3. **Clustering Analysis**: Proceed to KMeans and DBSCAN comparison as planned.
