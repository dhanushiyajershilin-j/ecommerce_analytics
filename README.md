# E-Commerce Analytics Flow
### Trend Analysis + Customer Segmentation + Behavioral Analysis

A complete, end-to-end implementation of the analytics flow: data collection →
cleaning → feature engineering → EDA → ML-based customer segmentation →
behavioral analysis → trend analysis → business insights → interactive dashboard.

## Project structure

```
ecommerce_analytics/
├── generate_dataset.py     # Step 1: creates a realistic synthetic dataset (no real data needed)
├── analytics_pipeline.py   # Steps 2-8: cleaning, RFM, feature engineering, clustering, trends, insights
├── dashboard.py             # Step 9: Streamlit interactive dashboard
├── requirements.txt
├── data/                     # raw generated data (created after running generate_dataset.py)
│   ├── customers.csv
│   ├── products.csv
│   ├── orders.csv
│   └── website_sessions.csv
└── outputs/                  # processed analytics outputs (created after running analytics_pipeline.py)
    ├── customer_features_segmented.csv
    ├── segment_summary.csv
    ├── monthly_trend.csv
    ├── weekday_trend.csv
    ├── category_monthly_trend.csv
    ├── new_customers_trend.csv
    ├── quarterly_trend.csv
    ├── business_insights.json
    └── summary.json
```

## Setup

```bash
pip install -r requirements.txt
```

## How to run

**Step 1 — generate the dataset** (skip this if you have your own real dataset —
see "Using your own data" below):
```bash
python generate_dataset.py
```

**Step 2 — run the analytics pipeline** (cleaning, RFM, feature engineering,
clustering, trend analysis, insights):
```bash
python analytics_pipeline.py
```

**Step 3 — launch the dashboard:**
```bash
streamlit run dashboard.py
```
It will open automatically in your browser at `http://localhost:8501`.
(The dashboard also auto-runs steps 1 & 2 the first time if `data/` or
`outputs/` are missing, so `streamlit run dashboard.py` alone works too.)

## What each script does

### `generate_dataset.py`
Creates 4 realistic CSV tables (matches "Sources" in the flow diagram):
- **customers.csv** — customer_id, name, gender, age, city, state, signup_date
- **products.csv** — product_id, category, sub_category, unit_price
- **orders.csv** — order-level transactions with quantity, discount, payment
  method, delivery days, order status (Delivered / Returned / Cancelled)
- **website_sessions.csv** — clickstream/behavior data (pages viewed, session
  duration, device, conversion)

Includes realistic seasonality (festive-season spikes in Oct/Nov/Dec),
year-over-year growth trend, and deliberately injected messiness (duplicates,
missing values) so the cleaning step has real work to do.

### `analytics_pipeline.py`
Implements steps 2–8 of the flow diagram:
1. **Cleaning** — dedup, missing-value imputation, cancellations excluded from
   revenue, time features created (year/month/week/weekday/quarter)
2. **Feature Engineering** — RFM (Recency, Frequency, Monetary), Average Order
   Value, basket size, purchase interval, discount usage rate, category
   diversity, return rate, web engagement features
3. **EDA** — revenue/orders summary stats, category & payment-method breakdowns
4. **Segmentation** — StandardScaler + log-transform → K-Means (k chosen via
   elbow method / silhouette score, mapped to 5 business-friendly segments:
   High Value, New, Price Sensitive, Occasional, At Risk/Inactive)
5. **Behavioral Analysis** — per-segment stats: AOV, basket size, discount
   usage, return rate, preferred category, conversion rate
6. **Trend Analysis** — monthly/weekly/quarterly revenue & order trends,
   category trends, new-customer growth, seasonal impact
7. **Business Insights** — auto-generated, data-driven recommendations

### `dashboard.py`
4-tab Streamlit dashboard with sidebar filters (date range, category, segment):
- **Trend Analysis** — monthly revenue, orders, weekday pattern, category
  trend over time, customer growth, seasonal/quarterly impact
- **Customer Segmentation** — segment pie chart, recency-vs-spend scatter,
  full segment profile table, RFM box plots, elbow/silhouette explainer
- **Behavioral Analysis** — AOV/discount/return-rate/category-diversity by
  segment, preferred category breakdown, weekday purchase radar
- **Business Insights** — auto-generated recommendations, segment radar
  comparison, revenue-vs-customer-share scatter (80/20 view)

## Using your own real dataset instead

If you have a real e-commerce dataset (e.g. from Kaggle, or your
company/college project data), replace the files in `data/` with your own,
keeping the same column names used above (or edit the column names at the top
of `analytics_pipeline.py`), then just run:
```bash
python analytics_pipeline.py
streamlit run dashboard.py
```
Everything downstream (RFM, clustering, trends, dashboard) works unchanged.

## Tech stack
`pandas` · `numpy` · `scikit-learn` (KMeans, StandardScaler, silhouette_score)
· `streamlit` · `plotly`
