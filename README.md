# 💼 SaaS Customer Intelligence & Analytics Framework

> An end-to-end AI-powered analytics platform for SaaS businesses — covering churn prediction, revenue forecasting, customer segmentation, and risk/opportunity scoring — with an interactive Streamlit dashboard for strategic decision-making.

---

## 📌 Project Overview

This project builds a complete **Customer Intelligence Pipeline** for a SaaS company using real-world business data. It ingests raw transactional, usage, and engagement data, transforms it into a unified master dataset, applies machine learning models to predict customer churn and future revenue, segments customers into behavioral cohorts, and presents all insights through an executive-grade interactive dashboard.

The end goal is to help Customer Success, Sales, and Revenue teams prioritize their accounts using data-driven scoring — identifying which customers to **Protect**, **Grow**, **Nurture**, or simply **Monitor**.

---

## 🗂️ Project Structure

```
├── data/                                        # Raw source CSV files (7 datasets)
│   ├── fact_customers.csv
│   ├── fact_usage_monthly.csv
│   ├── fact_transactions.csv
│   ├── fact_engagement_events.csv
│   ├── dim_geography.csv
│   ├── dim_industry.csv
│   └── dim_product.csv
│
├── output/                                      # Intermediate and final processed files
│   ├── master_dataset1.csv
│   └── cleaned_master_dataset.csv
│
├── config.py                                    # PostgreSQL connection configuration
├── data_integration.ipynb                       # Data ingestion, aggregation & feature engineering
├── data_cleaning.ipynb                          # Missing value handling & data quality
├── intelligent_customer_segmentation.ipynb      # KMeans clustering & cohort labeling
├── chrun_pred.ipynb                             # Churn probability model (classification)
├── revenue_pred.ipynb                           # Revenue forecasting model (regression)
├── risk_opportunity_scoring.ipynb               # Final intelligence profile assembly
├── app.py                                       # Streamlit dashboard application
└── README.md
```

---

## 🔄 Pipeline Architecture

```
Raw CSVs (7 tables)
      │
      ▼
[data_integration.ipynb]
  → Load into PostgreSQL
  → Aggregate usage, transactions, engagement
  → Merge into master_dataset (5272 rows × 108 cols)
      │
      ▼
[data_cleaning.ipynb]
  → Handle missing values
  → Engineer date features (contract duration, days to expiry)
  → Encode binary/categorical columns
  → Save: cleaned_master_dataset.csv
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
[chrun_pred.ipynb]               [revenue_pred.ipynb]
  → Classification model           → Regression model
  → Output: Churn_Probability       → Output: Predicted_Revenue
  → customer_churn_probability.csv  → customer_revenue_prediction.csv
      │                                      │
      └──────────────┬───────────────────────┘
                     ▼
      [intelligent_customer_segmentation.ipynb]
        → KMeans clustering on behavioral features
        → Output: customer_segments_labeled.csv
                     │
                     ▼
      [risk_opportunity_scoring.ipynb]
        → Merges churn + revenue + segments
        → Assigns Risk_Level & Revenue_Category
        → Output: customer_intelligence_profile.csv
                     │
                     ▼
              [app.py — Streamlit Dashboard]
        → Strategic Matrix, Segment Breakdown, Account Explorer
```

---

## 📊 Datasets

| Table | Rows | Description |
|---|---|---|
| `fact_customers` | 5,150 | Core customer attributes, ACV, tenure, contract info |
| `fact_usage_monthly` | 131,268 | Monthly product usage snapshots per customer |
| `fact_transactions` | 54,858 | Billing and revenue transaction records |
| `fact_engagement_events` | 54,822 | Support tickets, NPS scores, escalations |
| `dim_geography` | 183 | Country/region lookup |
| `dim_industry` | 22 | Industry vertical lookup |
| `dim_product` | 32 | Product/plan tier lookup |

---

## 🧠 ML Models

### 1. Churn Prediction (Classification)
- **Target:** `Churn` (binary)
- **Key Features:** Tenure, DAU trend, feature adoption %, payment delays, NPS score, escalation history, renewal risk flag
- **Techniques:** Feature selection via correlation, binary/categorical encoding, StandardScaler, sklearn Pipeline
- **Output:** `Churn_Probability` per customer (0–1)

### 2. Revenue Forecasting (Regression)
- **Target:** `Next_Quarter_Revenue_USD`
- **Key Features:** ACV, transaction history, add-on revenue, usage trends, contract progression
- **Statistical Testing:** Pearson correlation for numerical features; ANOVA/T-test for categorical features
- **Output:** `Predicted_Revenue` per customer

### 3. Customer Segmentation (Clustering)
- **Algorithm:** KMeans with optimal k selected via Elbow Method + Silhouette Score
- **Preprocessing:** Median imputation for numerics, One-Hot Encoding for categoricals, StandardScaler
- **Dimensionality Reduction:** PCA for visualization
- **Leakage Prevention:** Target columns (`Churn`, `Next_Quarter_Revenue_USD`, `Renewal_Probability`) excluded from clustering features
- **Output:** `Cluster_Label` + descriptive `Cluster_Name` per customer

---

## 🎯 Strategic Segmentation Logic

Each customer is assigned one of four **Action Segments** in the dashboard:

| Segment | Condition | Recommended Action |
|---|---|---|
| 🔴 **PROTECT** | Churn Risk ≥ 50% | Immediate retention intervention |
| 🟢 **GROW** | Revenue Opportunity Score ≥ 40% | Upsell / expansion outreach |
| 🟡 **NURTURE** | Mid-range on both axes | Proactive engagement campaigns |
| 🔵 **MONITOR** | Low risk AND low opportunity | Automated health check cadence |

---

## 🖥️ Dashboard Features (`app.py`)

Built with **Streamlit** and **Plotly**, the dashboard provides:

- **KPI Cards** — Active accounts, at-risk count, total run-rate ACV, AI-predicted revenue
- **Strategic Matrix** — Scatter plot of Churn Risk vs Revenue Opportunity, sized by predicted revenue
- **Segment Breakdown** — Donut chart for quadrant distribution; grouped bar chart comparing Current ACV vs Predicted Revenue by cluster
- **Account Explorer** — Sortable data table for daily triage workflows

**Sidebar Filters:**
- Priority strategy segment selection
- AI customer cohort cluster selection
- Minimum churn probability threshold slider

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- PostgreSQL (local or remote)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/saas-customer-intelligence.git
cd saas-customer-intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Core packages:**
```
pandas, numpy, scikit-learn, matplotlib, seaborn,
sqlalchemy, psycopg2-binary, streamlit, plotly, joblib, scipy
```

### 3. Configure Database
Edit `config.py` or set environment variables:
```bash
export PG_USER=postgres
export PG_PASS=your_password
export PG_HOST=localhost
export PG_PORT=5432
export PG_DB=Final_Project_DB
```

### 4. Run the Pipeline (in order)
```bash
jupyter nbconvert --to notebook --execute data_integration.ipynb
jupyter nbconvert --to notebook --execute data_cleaning.ipynb
jupyter nbconvert --to notebook --execute intelligent_customer_segmentation.ipynb
jupyter nbconvert --to notebook --execute chrun_pred.ipynb
jupyter nbconvert --to notebook --execute revenue_pred.ipynb
jupyter nbconvert --to notebook --execute risk_opportunity_scoring.ipynb
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 🔑 Key Feature Engineering

| Feature | Description |
|---|---|
| `dau_trend` | Linear slope of Daily Active Users over time (engagement momentum) |
| `payment_reliability_score` | 100 minus average payment delay days |
| `avg_monthly_revenue` | Total net revenue divided by tenure months |
| `revenue_per_employee` | ACV normalized by company headcount |
| `licenses_unused` | Licenses purchased minus actual active users |
| `contract_progress` | % of contract period elapsed |
| `days_to_expiry` | Days remaining until contract end |
| `near_renewal` | Binary flag: contract expiring within 30 days |
| `positive_sentiment_ratio` | % of engagement events with positive sentiment |

---

## 📈 Output Files

| File | Contents |
|---|---|
| `master_dataset1.csv` | Merged raw dataset (5272 rows, 108 cols) |
| `cleaned_master_dataset.csv` | Cleaned, encoded, feature-engineered dataset |
| `customer_churn_probability.csv` | Customer ID + Churn_Probability |
| `customer_revenue_prediction.csv` | Customer ID + Predicted_Revenue |
| `customer_segments_labeled.csv` | Customer ID + Cluster_Label + Cluster_Name |
| `customer_intelligence_profile.csv` | Unified profile (input to dashboard) |

---

## 🛡️ Data Quality Practices

- Duplicate removal at every pipeline stage
- Rule-based imputation for `Company_Size` using `Employees` count before mode-fill fallback
- Correlation-based feature pruning (threshold > 0.85) to reduce multicollinearity in clustering
- Target leakage prevention: churn/revenue targets explicitly excluded from segmentation features
- Outer merge deduplication by `Customer_ID` in the dashboard layer

---

## 📋 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
sqlalchemy>=2.0
psycopg2-binary>=2.9
streamlit>=1.30
plotly>=5.18
joblib>=1.3
```

---

## 👤 Author

**Nithyanantham G**  
Data Science & Analytics  
[LinkedIn](#) | [GitHub](#)

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
