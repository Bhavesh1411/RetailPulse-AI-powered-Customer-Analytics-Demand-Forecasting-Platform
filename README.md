# RetailPulse: AI-Powered Retail Analytics & Decision Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3949AB?style=flat)](https://github.com/microsoft/LightGBM)
[![XGBoost](https://img.shields.io/badge/XGBoost-2C3E50?style=flat)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=Plotly&logoColor=white)](https://plotly.com/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Project Status](https://img.shields.io/badge/Status-Pilot%20Ready-green.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#)

> **Academic Notice:** This repository was developed as an academic capstone project.

RetailPulse is an end-to-end predictive machine learning and decision intelligence platform built for B2B wholesale retail operators. By combining advanced classification, regression, and clustering algorithms with operations research optimization heuristics, the platform turns raw transaction ledgers into proactive strategies for customer retention, inventory management, demand readiness, and automated risk mitigation.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Platform Architecture](#2-platform-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Technology Stack](#4-technology-stack)
5. [Machine Learning & Heuristic Modules](#5-machine-learning--heuristic-modules)
6. [Interactive Dashboard Modules](#6-interactive-dashboard-modules)
7. [Operational Workflow](#7-operational-workflow)
8. [Performance & Business Results](#8-performance--business-results)
9. [Installation & Setup](#9-installation--setup)
10. [Repository Details](#10-repository-details)
11. [Repository Statistics](#11-repository-statistics)

---

## 1. Project Overview

### The Business Problem
Wholesale B2B retail operations suffer from fragmented decision-making:
* **Customer Churn:** Managers struggle to identify when commercial accounts are drifting away before they completely stop ordering.
* **Stockouts vs. Holding Costs:** Over-purchasing ties up working capital, while stockouts of high-velocity items cause massive revenue drops.
* **Unpredictable Demand:** Extreme seasonal fluctuations (like Q4 holiday spikes) lead to capacity bottlenecks or stock gluts.
* **Generic Marketing:** Treating small weekend buyers and massive weekday enterprise clients the same way dilutes marketing ROI.

### Platform Solution
RetailPulse solves these operational pain points by integrating predictive ML models and inventory heuristics into a single, cohesive, secure decision-support dashboard. Instead of generating abstract scores, it prescribes concrete actions—such as specific purchase orders (EOQs) and personalized retention sequences—tailored to each product class and customer risk cohort.

---

## 2. Platform Architecture

The unidirectional pipeline maps data ingestion from raw files through processing, modeling, and output stages, eventually presenting interactive strategies to the user:

```mermaid
graph TD
    A[Raw Data] --> B[Data Cleaning]
    B --> C[Feature Engineering]
    C --> D[Machine Learning Models]
    D --> E[Processed Outputs]
    E --> F[Dashboard]
    F --> G[Business Decision Support]
```

---

## 3. Folder Structure

The repository follows a clean, modular structure separating machine learning pipelines, precomputed datasets, static analysis reports, and dashboard pages:

```text
RetailPulse/
├── churn_prediction_true/     # True Churn Prediction Pipeline
│   ├── models/                # Serialized model binaries (.pkl)
│   ├── predictions/           # Generated batch inference outputs (.csv)
│   ├── reports/               # Churn-specific evaluation markdown reports
│   └── scripts/               # Data prep, training, and report generation scripts
├── customer_segmentation/     # Customer Segmentation Module
│   ├── datasets/              # Segment assignment files (.csv)
│   ├── documentation/         # Feature definitions & pipeline steps
│   ├── reports/               # KMeans profiling & DBSCAN validation reports
│   └── scripts/               # Clustering, cleaning, and evaluation scripts
├── dashboard/                 # Streamlit Web Application
│   ├── app.py                 # Multi-page landing entrypoint
│   ├── auth/                  # Authentication & session state management
│   ├── components/            # Sidebar navigation & common UI wrappers
│   └── pages/                 # Streamlit UI modules (Pages 1 to 10)
├── Demand_Forecasting/        # Weekly Demand Forecasting Module
│   ├── datasets/              # Weekly forecast datasets (.json, .csv)
│   ├── reports/               # Hyperparameter tuning and MAPE reports
│   └── scripts/               # Optuna tuning, lag engineering, training scripts
├── eda_output/                # EDA visualization output files
├── inventory_optimization/    # Inventory Optimization Engine
│   ├── outputs/               # Master reorder matrices (.csv)
│   └── reports/               # ABC & EOQ heuristic summaries (.md, .json)
├── processed_data/            # System-wide cleaned datasets & audit summaries
├── requirements.txt           # Dependency management mapping
└── README.md                  # Primary repository documentation
```

---

## 4. Technology Stack

| Category | Technologies Used |
| :--- | :--- |
| **Programming Language** | Python (v3.11) |
| **Machine Learning** | Scikit-Learn (v1.8.0), LightGBM (v4.6.0), XGBoost (v3.2.0) |
| **Data Processing** | Pandas (v3.0.1), NumPy (v2.4.6) |
| **Visualization** | Plotly (v6.8.0), Matplotlib (v3.10.8), Seaborn (v0.13.2) |
| **Dashboard** | Streamlit (v1.56.0) |
| **Storage** | CSV, JSON |
| **Version Control** | Git |

---

## 5. Machine Learning & Heuristic Modules

### Customer Segmentation
* **Algorithm:** KMeans Clustering (Optimal $K=3$ selected via Silhouette and Elbow validation; refund/cancellation outliers representing 103 records or 2.35% of the base were systematically cleaned).
* **Feature Matrix:** Recency, Frequency, Monetary (RFM), Customer Tenure, Average Order Value (AOV), and Weekend Sales Ratio.
* **Target Cohorts Identified:**
  * **VIP Loyal Champions:** 1,537 accounts (35.9% of base). Drive 85% of total company revenue ($7.04M spend). Average recency of 35.7 days.
  * **Lapsing Occasional Buyers:** 2,116 accounts (49.5% of base). High-risk inactive group averaging 123.8 days since their last order.
  * **Weekend Value Shoppers:** 623 accounts (14.6% of base). A distinct niche segment that conducts 76% of transactions over weekends.
* **Business Application:** Recommends concierge services for the top-tier VIP champions, reactivation win-back email cycles (at 90/120/150 day boundaries) for lapsing buyers, and Thursday evening promotions for weekend shoppers.

### True Churn Prediction
* **Algorithm:** LightGBM Binary Classifier fitted with class weights to compensate for target imbalance.
* **Target Construct:** Predicts whether a customer will record zero orders over a forward-looking 90-day window based strictly on chronological snapshots up to a cutoff point (`2010-09-10`).
* **Evaluation Performance (Holdout Set):**
  * **ROC AUC:** `71.77%`
  * **Recall (Sensitivity):** `62.31%`
  * **Accuracy:** `65.55%`
  * **Precision:** `61.07%`
  * **F1-Score:** `61.68%`
* **Core Risk Levers:** Identified through SHAP value attribution. *Recency* and *Average Days Between Purchases* are the highest risk indicators, while *Quantity Per Order*, *AOV*, and *Customer Tenure* act as strong retention buffers.
* **Business Application:** Identifies and ranks customers with >70% churn probability, triggering high-priority outreach before attrition occurs.

### Demand Forecasting
* **Algorithm:** Tuned XGBoost Regressor incorporating Calendrical Holiday Features.
* **Granularity:** Weekly aggregated sales (chosen over daily models to isolate high-frequency logistics noise).
* **Validation Performance (Chronological Split):**
  * **Validation MAPE:** `11.81%` (Successfully achieved target mandate of $\le 12\%$).
  * **Test MAPE:** `16.97%` (Test MAPE rises to `17.60%` when excluding the final partial week).
* **Tuning Protocol:** 100 trials of Optuna parameter optimization. Top parameters: `n_estimators=451`, `learning_rate=0.052`, `max_depth=5`, `subsample=0.888`.
* **Key Engineered Features:** `holiday_proximity_score` (representing 25.24% of feature importance), `weeks_until_christmas` (14.06%), `weeks_since_christmas` (10.14%), and rolling sales lags.

### Inventory Optimization
* **Methodology:** Operations Research heuristics utilizing demand statistics calculated over the verified calendar window of **699 days** (`2009-01-12` to `2010-12-11`).
* **ABC Classification (Revenue Pareto Principle):**
  * **Class A:** 889 products (20.9% of catalog). Generates ~80% of revenue. Set to 99% Service Level (Z=2.33, Lead Time=7 days).
  * **Class B:** 1,025 products (24.0% of catalog). Generates ~15% of revenue. Set to 95% Service Level (Z=1.65, Lead Time=10 days).
  * **Class C:** 2,348 products (55.1% of catalog). Generates ~5% of revenue. Set to 90% Service Level (Z=1.28, Lead Time=14 days).
* **Safety Stock & ROP Formulas:**

  $$SS = Z \times \sigma_{\text{daily demand}} \times \sqrt{LT}$$

  $$ROP = (d_{\text{daily demand}} \times LT) + SS$$

  $$EOQ = \sqrt{\frac{2 \times \text{Annual Demand} \times \text{Order Cost (\$50.00)}}{\text{Holding Cost (20\% of Unit price)}}}$$
* **Current Operations Metrics:** Evaluated catalog health yields an **Inventory Health Score of 52.44/100** (🔴 Critical Attention Required) due to simulated stock falling below safety limits on 36% of the catalog.

---

## 6. Interactive Dashboard Modules

The user interface is split into 10 key modules, protected behind session-based authentication:

| Page | Purpose | Key Features |
| :--- | :--- | :--- |
| `app.py` | **Authentication Gateway** | Secure login portal handling session parameters and routing validations. |
| `1_Executive_Overview.py` | **Executive Overview** | High-level platform index summarizing operational capabilities. |
| `2_Demand_Forecasting.py` | **Demand Forecasting** | Out-of-sample weekly demand projections (8-week horizon) with validation MAPE metrics. |
| `3_Customer_Segmentation.py` | **Customer Segmentation** | Customer segment shares (donut chart), RFM matrices, and strategic playbook tabs. |
| `4_Churn_Prediction.py` | **Churn Prediction** | Customer risk distribution charts, SHAP importance bars, and high-probability attrition list. |
| `5_Inventory_Optimization.py` | **Inventory Optimization** | ABC revenue pie charts, stock risk bar charts, and immediate purchase recommendations. |
| `6_Alerts_and_Monitoring.py` | **Alerts & Monitoring** | Live warning hub highlighting inventory deficits, revenue at risk, and demand surges/drops. |
| `7_Export_Center.py` | **Export Center** | Tabular export suite for raw/processed prediction CSVs and Excel reports. |
| `8_Admin_Panel.py` | **Admin Panel** | Configuration controls for thresholds, lead times, and service levels. |
| `9_Prediction_and_Decision_Center.py` | **Prediction & Decision Center** | Joint lookup querying individual Customer IDs and Product IDs to prescribe actions. |
| `10_Model_Registry.py` | **Model Registry** | Status center checking if ML model assets exist and displaying training specifications. |

---

## 7. Operational Workflow

```text
               [STAGE 1: CLEANING]
               - Ingest transaction records (Initial: 525,461 rows)
               - Remove duplicates (6,865 rows) and bad debt/test records
               - Final cleaned baseline: 514,834 rows
                        │
                        ▼
               [STAGE 2: FEATURE ENGINEERING]
               - Aggregate transactions to weekly timelines for demand lag structures
               - Calculate recency, frequency, monetary, and tenure values per customer
                        │
                        ▼
               [STAGE 3: MODEL TRAINING & TUNING]
               - Run KMeans on customer features to segment the buyer base
               - Train LightGBM classifier with class weights to predict 90-day churn
               - Train Optuna-optimized XGBoost regressor for demand forecasting
                        │
                        ▼
               [STAGE 4: HEURISTIC FORMULATION]
               - Perform revenue sorting to categorize catalog items (Class A/B/C)
               - Calculate EOQ, Safety Stock, and ROP buffers over a 699-day window
                        │
                        ▼
               [STAGE 5: DECISION PRESENTATION]
               - Serve outputs through Streamlit pages (using precomputed CSV/JSON datasets)
               - Automatically surface critical warnings via the Alerts and Monitoring system
```

---

## 8. Performance & Business Results

| Module | Algorithm | Primary Metric | Business Outcome |
| :--- | :--- | :--- | :--- |
| **Customer Segmentation** | KMeans Clustering | Silhouette Score: `0.3250` | Identified 3 distinct behavioural cohorts; VIP segment drives 85% of revenue ($7.04M spend). |
| **True Churn Prediction** | LightGBM Classifier | ROC AUC: `71.77%`, Recall: `62.31%`, Accuracy: `65.55%` | Flags high-risk churn customers (actual future churn rate of 91.58%) for proactive outreach. |
| **Demand Forecasting** | XGBoost Regressor | Validation MAPE: `11.81%`, Test MAPE: `16.97%` | Provides 8-week sales projection capturing Q4 holiday restocking surges. |
| **Inventory Heuristics** | ABC & EOQ Engine | Inventory Health Score: `52.44/100` | Classifies catalog, calculates Safety Stock/ROP/EOQ, and flags 36% of catalog at critical risk. |

---

## 9. Installation & Setup

### Prerequisites
* Python 3.11
* Virtualenv (Recommended)

### Step-by-Step Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bhavesh1411/RetailPulse-AI-powered-Customer-Analytics-Demand-Forecasting-Platform.git
   cd RetailPulse-AI-powered-Customer-Analytics-Demand-Forecasting-Platform
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install platform requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the interactive Streamlit dashboard:**
   ```bash
   cd dashboard
   streamlit run app.py
   ```

5. **Platform Authentication:**
   * Demo credentials are available for authorized evaluators or can be configured locally.

---

## 10. Repository Details

* **Project Status:** Pilot Ready / Business Validation Complete
* **Current Version:** v1.0.0
* **Author:** Bhavesh1411
* **License:** MIT License
* **Last Updated:** 2026-06-30
* **Repository Purpose:** Comprehensive decision-support system converting retail transaction logs into retention, demand, and inventory strategies.
* **Disclaimer:** Inventory stock balances are generated as simulations using a reproducible seed (seed=42) to support decision-making, as historical stock values were absent from the transaction logs. All outputs should be used as decision-support simulations.

---

## 11. Repository Statistics

The following statistics summarize the current state of the implementation *(Note: These statistics reflect the state of the repository at the time of generation)*:

* **Total Dashboard Pages:** 10
* **Total ML Models:** 3 (KMeans Clustering, LightGBM Classifier, XGBoost Regressor)
* **Total Reports:** 19
* **Total Datasets:** 10
* **Total Prediction Outputs:** 3
* **Total Inventory Reports:** 1