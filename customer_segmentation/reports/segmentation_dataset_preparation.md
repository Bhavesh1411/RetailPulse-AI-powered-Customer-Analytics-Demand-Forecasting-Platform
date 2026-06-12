# Customer Segmentation Dataset Preparation

This report details the steps taken to prepare the approved feature set for clustering. We have processed the data, analyzed skewness, applied necessary transformations, and scaled the dataset.

## 1. Data Cleaning
- **Initial Shape**: `(4379, 8)`
- **Null Values**: 0 nulls detected.
- **Infinite Values**: Checked and replaced with NaN. No infinite values were dropped.
- **Final Shape**: `(4379, 8)`

## 2. Feature Distribution Summary

We analyzed the skewness of the approved features plus `active_months`. A skewness absolute value $> 1.0$ is generally considered highly skewed and an excellent candidate for log transformation.

| Feature | Skewness (Before) | Distribution Profile |
| :--- | :---: | :--- |
| **customer_tenure** | `-0.4054` | **Normal-ish**. No transformation needed. |
| **recency** | `1.2736` | **Moderately Skewed**. Right-tailed (many recent customers, few very old ones). |
| **active_months** | `1.5644` | **Moderately Skewed**. Right-tailed (most customers active for 1-3 months). |
| **weekend_sales_ratio** | `2.0029` | **Highly Skewed**. Many customers with 0% weekend sales. |
| **frequency** | `10.5623` | **Extremely Skewed**. A few customers have massive order counts. |
| **average_purchase_value** | `13.6237` | **Extremely Skewed**. A few customers have massive basket sizes. |
| **monetary** | `24.8400` | **Extremely Skewed**. Classic Pareto distribution (80/20 rule) of spend. |

## 3. Log Transformation Report

To prevent K-Means from forming outlier-only clusters (where a single massive spender becomes their own cluster), we applied a `log1p` transformation ($log(x + 1)$) to all features with initial skewness $> 1.0$.

| Feature | Skewness (Before) | Skewness (After log1p) | Transformation Effect |
| :--- | :---: | :---: | :--- |
| `recency` | 1.2736 | **-0.4350** | Fixed. Now approximately symmetric. |
| `active_months` | 1.5644 | **0.6297** | Fixed. Reduced to mild skew. |
| `frequency` | 10.5623 | **1.0216** | Greatly improved. Skewness reduced by 90%. |
| `monetary` | 24.8400 | **-1.2072** | Greatly improved. Massive outliers pulled in. |
| `weekend_sales_ratio` | 2.0029 | **1.7681** | Slightly improved. (Skew persists due to zero-inflation). |
| `average_purchase_value`| 13.6237 | **-2.5668** | Greatly improved. Shifted from right-skew to moderate left-skew. |

> [!TIP]
> The log transformation successfully compressed the extreme right tails of `monetary`, `frequency`, and `average_purchase_value`, making them much more suitable for distance-based algorithms like KMeans.

## 4. Scaling Comparison (Standard vs. Robust)

After log-transforming, we compared two scaling strategies to ensure all features contribute equally to the Euclidean distance calculations.

### **StandardScaler**
- **Mechanism**: Subtracts the mean and divides by standard deviation.
- **Result**: Every feature has exactly Mean = 0 and Variance = 1.
- **Advantage**: Perfectly equalizes variance across all features, which is exactly what KMeans Euclidean distance assumes.

### **RobustScaler**
- **Mechanism**: Subtracts the median and divides by the Interquartile Range (IQR).
- **Result**: Every feature has exactly Median = 0 and IQR = 1.
- **Advantage**: Ignores outliers completely when scaling.
- **Disadvantage**: Features end up with different absolute variances, meaning features with long tails (like our log-transformed `average_purchase_value`) will exert more pull on the KMeans centroids than tightly clustered features.

### **Selected Strategy: StandardScaler**
We justify choosing **StandardScaler** because we have *already* handled the extreme outliers using the Log Transformation. 
Because K-Means calculates squared Euclidean distances, giving every feature an equal variance ($\sigma^2 = 1$) via StandardScaler ensures equal geometric weighting across all 7 dimensions. RobustScaler would leave unequal variances, unintentionally biasing the clusters.

---

## Final Clustering Dataset

The final processed dataset has been successfully saved to:
`C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\clustering_dataset_std.csv`

**Features Included**: `customer_id` (identifier), `recency` (scaled, log), `frequency` (scaled, log), `monetary` (scaled, log), `average_purchase_value` (scaled, log), `customer_tenure` (scaled, raw), `weekend_sales_ratio` (scaled, log), and `active_months` (scaled, log).

> [!IMPORTANT]
> **User Review Required**:
> We are now ready to run KMeans and DBSCAN clustering using this standardized dataset. We will run comparisons both with and without `active_months` as requested. 
> 
> Please provide your approval to proceed to the Model Execution phase.
