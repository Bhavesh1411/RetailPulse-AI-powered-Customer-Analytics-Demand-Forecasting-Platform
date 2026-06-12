# Cleaned KMeans Clustering Evaluation

Following the audit of Cluster 1, we excluded customers dominated by refunds and reran the entire clustering pipeline (log transformations, standard scaling, and KMeans evaluation).

## 1. Refund Audit & Exclusion

* **Total Original Customers**: 4,379
* **Number of Customers Removed**: **103**
* **Percentage of Total Base**: 2.35%
* **Profile of Excluded Customers**:
  * Median Monetary Value: $0.00
  * Median Quantity Per Order: -1.00
* **Reason for Exclusion**: These customers had artificially identical profiles (zero monetary values and negative order quantities) due to refund/cancellation activity. They formed an artificial dense micro-cluster that distorted the KMeans algorithm.

## 2. New Dataset & Model Metrics

* **New Customer Count**: **4,276**
* **New Clustering Dataset Size**: (4276, 6 features)
* **New Optimal K**: **3** (Based on Silhouette Score)
* **New Silhouette Score**: **0.2898**

## 3. New Cluster Sizes & Profiles (Original Space)

The removal of the anomaly group cleanly reduced our optimal K from 4 down to 3. The remaining clusters perfectly capture the three genuine, actionable customer behaviors we identified previously.

| Cluster | Count | % of Base | recency | frequency | monetary | avg_purchase | tenure | weekend_ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | 1,537 | 35.9% | 35.7 days | 11.29 | $4,580.55 | $403.38 | 306.6 days | 11% |
| **Cluster 1** | 2,116 | 49.5% | 123.8 days| 1.96 | $452.45 | $262.35 | 177.8 days | 0% |
| **Cluster 2** | 623 | 14.6% | 99.6 days | 2.92 | $660.55 | $249.14 | 208.0 days | 76% |

## 4. Comparison with Previous Clustering

| Metric | Previous Model (With Anomalies) | New Cleaned Model |
| :--- | :--- | :--- |
| **Optimal K** | 4 | **3** |
| **Silhouette Score** | 0.3250 | **0.2898** |
| **Anomaly Cluster Present?** | Yes (Cluster 1: 118 customers) | **No** |
| **Genuine Segments Preserved?** | Yes | **Yes** (Identical behavioral profiles) |

* **Why did the Silhouette Score drop slightly?** The silhouette score measures cluster tightness. Because the anomaly group (103 customers) was incredibly dense (almost all had identically $0 and -1 quantities), it artificially inflated the global silhouette score. The new score of 0.2898 is a genuine reflection of natural behavioral boundaries without the mathematical distortion of the outliers.

The new cleaned cluster assignments have been securely saved to:
`C:\Users\LENOVO\OneDrive\Desktop\RetailPulse\processed_data\customer_segments_kmeans_v2.csv`
