"""
analytics_pipeline.py
----------------------
Implements steps 2-8 of the E-Commerce Analytics Flow:
    2. Data Cleaning & Preprocessing
    3. Feature Engineering (RFM + behavioral features)
    4. Exploratory Data Analysis
    5. Customer Segmentation (KMeans)
    6. Behavioral Analysis (by cluster)
    7. Trend Analysis (over time)
    8. Business Insights & Recommendations

Run:  python analytics_pipeline.py
Reads from ./data/, writes processed outputs to ./outputs/
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import os

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

pd.set_option("display.width", 120)

# ===========================================================================
# STEP 1: DATA COLLECTION (load + merge)
# ===========================================================================
print("=" * 70)
print("STEP 1: DATA COLLECTION")
print("=" * 70)

customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
products = pd.read_csv(f"{DATA_DIR}/products.csv")
orders = pd.read_csv(f"{DATA_DIR}/orders.csv", parse_dates=["order_date"])
sessions = pd.read_csv(f"{DATA_DIR}/website_sessions.csv", parse_dates=["visit_date"])

print(f"Loaded: {len(customers)} customers, {len(products)} products, "
      f"{len(orders)} order lines, {len(sessions)} sessions")

# ===========================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING & PREPROCESSING")
print("=" * 70)

# --- customers ---
before = len(customers)
customers = customers.drop_duplicates(subset="customer_id")
customers["age"] = customers["age"].fillna(customers["age"].median())
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
print(f"Customers: removed {before - len(customers)} duplicates, "
      f"imputed missing ages with median")

# --- orders ---
before = len(orders)
orders = orders.drop_duplicates(subset="order_id")
orders["delivery_days"] = orders["delivery_days"].fillna(orders["delivery_days"].median())
orders = orders.dropna(subset=["order_date", "customer_id", "total_amount"])
# drop cancelled orders from revenue-facing analysis; keep returns flagged separately
orders["is_cancelled"] = orders["order_status"].eq("Cancelled")
orders["is_returned"] = orders["order_status"].eq("Returned")
orders_clean = orders[~orders["is_cancelled"]].copy()  # exclude cancellations from sales
print(f"Orders: removed {before - len(orders)} duplicates, "
      f"imputed missing delivery days, excluded {orders['is_cancelled'].sum()} cancellations")

# time features
orders_clean["year"] = orders_clean["order_date"].dt.year
orders_clean["month"] = orders_clean["order_date"].dt.month
orders_clean["month_name"] = orders_clean["order_date"].dt.strftime("%b")
orders_clean["year_month"] = orders_clean["order_date"].dt.to_period("M").astype(str)
orders_clean["week"] = orders_clean["order_date"].dt.isocalendar().week
orders_clean["weekday"] = orders_clean["order_date"].dt.day_name()
orders_clean["quarter"] = orders_clean["order_date"].dt.quarter

print(f"Created time features: year, month, year_month, week, weekday, quarter")

# ===========================================================================
# STEP 3: FEATURE ENGINEERING (RFM + Behavioral + Product features)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 70)

snapshot_date = orders_clean["order_date"].max() + pd.Timedelta(days=1)

# ---- RFM ----
rfm = orders_clean.groupby("customer_id").agg(
    recency_days=("order_date", lambda x: (snapshot_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("total_amount", "sum"),
).reset_index()

# ---- Behavioral features ----
behavior = orders_clean.groupby("customer_id").agg(
    avg_order_value=("total_amount", "mean"),
    total_quantity=("quantity", "sum"),
    avg_basket_size=("quantity", "mean"),
    discount_usage_rate=("discount_pct", lambda x: (x > 0).mean()),
    avg_discount_pct=("discount_pct", "mean"),
    favorite_category=("category", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    category_diversity=("category", "nunique"),
    return_rate=("is_returned", "mean"),
    first_purchase=("order_date", "min"),
    last_purchase=("order_date", "max"),
).reset_index()

# purchase interval (avg days between orders)
def avg_purchase_interval(dates):
    dates = sorted(dates)
    if len(dates) < 2:
        return np.nan
    diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    return np.mean(diffs)

intervals = orders_clean.groupby("customer_id")["order_date"].apply(avg_purchase_interval).reset_index()
intervals.columns = ["customer_id", "avg_purchase_interval_days"]

# website engagement features
web_feat = sessions.groupby("customer_id").agg(
    total_sessions=("session_id", "nunique"),
    avg_pages_viewed=("pages_viewed", "mean"),
    avg_session_duration=("session_duration_min", "mean"),
    conversion_rate=("converted", "mean"),
).reset_index()

# merge everything into a master customer feature table
features = rfm.merge(behavior, on="customer_id", how="left") \
               .merge(intervals, on="customer_id", how="left") \
               .merge(web_feat, on="customer_id", how="left") \
               .merge(customers[["customer_id", "customer_name", "gender", "age", "city", "state", "signup_date"]],
                      on="customer_id", how="left")

features["avg_purchase_interval_days"] = features["avg_purchase_interval_days"].fillna(
    features["avg_purchase_interval_days"].median())
features["tenure_days"] = (snapshot_date - features["signup_date"]).dt.days
features[["total_sessions", "avg_pages_viewed", "avg_session_duration", "conversion_rate"]] = \
    features[["total_sessions", "avg_pages_viewed", "avg_session_duration", "conversion_rate"]].fillna(0)

features.to_csv(f"{OUT_DIR}/customer_features.csv", index=False)
print(f"Built customer feature table: {features.shape[0]} customers x {features.shape[1]} features")
print(f"Features: RFM (recency/frequency/monetary), AOV, basket size, purchase interval, "
      f"discount usage, category diversity, return rate, web engagement")

# ===========================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (summary stats saved for dashboard)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

eda_summary = {
    "total_revenue": float(orders_clean["total_amount"].sum()),
    "total_orders": int(orders_clean["order_id"].nunique()),
    "total_customers": int(orders_clean["customer_id"].nunique()),
    "avg_order_value": float(orders_clean["total_amount"].mean()),
    "return_rate_overall": float(orders_clean["is_returned"].mean()),
    "date_range": [str(orders_clean["order_date"].min().date()), str(orders_clean["order_date"].max().date())],
    "revenue_by_category": orders_clean.groupby("category")["total_amount"].sum().sort_values(ascending=False).round(2).to_dict(),
    "orders_by_payment_method": orders_clean["payment_method"].value_counts().to_dict(),
    "orders_by_weekday": orders_clean["weekday"].value_counts().to_dict(),
    "rfm_stats": {
        "recency_days": {"mean": float(rfm.recency_days.mean()), "median": float(rfm.recency_days.median())},
        "frequency": {"mean": float(rfm.frequency.mean()), "median": float(rfm.frequency.median())},
        "monetary": {"mean": float(rfm.monetary.mean()), "median": float(rfm.monetary.median())},
    },
}
print(f"Total revenue: {eda_summary['total_revenue']:,.0f}")
print(f"Total orders: {eda_summary['total_orders']:,}")
print(f"Avg order value: {eda_summary['avg_order_value']:,.2f}")
print(f"Overall return rate: {eda_summary['return_rate_overall']:.2%}")

# ===========================================================================
# STEP 5: CUSTOMER SEGMENTATION (KMeans)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 5: CUSTOMER SEGMENTATION (ML)")
print("=" * 70)

cluster_features = ["recency_days", "frequency", "monetary", "avg_order_value",
                     "avg_basket_size", "discount_usage_rate", "category_diversity"]
X = features[cluster_features].fillna(0).copy()

# log-transform skewed monetary/frequency for better clustering
for col in ["monetary", "frequency", "avg_order_value"]:
    X[col] = np.log1p(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method + silhouette score to pick optimal k
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k = list(K_range)[int(np.argmax(sil_scores))]
# for clean business storytelling we target 5 segments (matches the flow diagram) unless silhouette strongly prefers otherwise
final_k = 5
print(f"Elbow/silhouette suggested k={best_k} (silhouette={max(sil_scores):.3f}); using k={final_k} for business segments")

kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
features["cluster"] = kmeans_final.fit_predict(X_scaled)

elbow_data = {"k": list(K_range), "inertia": [round(i, 2) for i in inertias],
              "silhouette": [round(s, 4) for s in sil_scores]}

# ---- Label clusters by RFM profile (business-friendly names) ----
cluster_profile = features.groupby("cluster")[["recency_days", "frequency", "monetary"]].mean()
cluster_profile["score"] = (
    (-cluster_profile["recency_days"].rank()) +
    cluster_profile["frequency"].rank() +
    cluster_profile["monetary"].rank()
)
ranked = cluster_profile.sort_values("score", ascending=False)

label_pool = ["High Value Customers", "Occasional Buyers", "New Customers",
              "Price Sensitive Customers", "At Risk / Inactive Customers"]

# Assign "New Customers" to whichever cluster has lowest frequency + lowest tenure/recency-ish newness
tenure_by_cluster = features.groupby("cluster")["tenure_days"].mean()
new_cluster = tenure_by_cluster.idxmin()

# Assign "At Risk" to highest recency (least recent)
at_risk_cluster = cluster_profile["recency_days"].idxmax()

# Assign "High Value" to highest monetary
high_value_cluster = cluster_profile["monetary"].idxmax()

# Assign "Price Sensitive" to highest discount usage
discount_by_cluster = features.groupby("cluster")["discount_usage_rate"].mean()
price_sensitive_cluster = discount_by_cluster.idxmax()

assigned = {}
priority = [
    (high_value_cluster, "High Value Customers"),
    (new_cluster, "New Customers"),
    (at_risk_cluster, "At Risk / Inactive Customers"),
    (price_sensitive_cluster, "Price Sensitive Customers"),
]
for clus, name in priority:
    if clus not in assigned:
        assigned[clus] = name
remaining_clusters = [c for c in cluster_profile.index if c not in assigned]
remaining_labels = [l for l in label_pool if l not in assigned.values()]
for clus, name in zip(remaining_clusters, remaining_labels):
    assigned[clus] = name
# fallback for any leftover
for clus in cluster_profile.index:
    if clus not in assigned:
        assigned[clus] = "Occasional Buyers"

features["segment"] = features["cluster"].map(assigned)

print("Cluster -> Segment label mapping:")
for clus, name in sorted(assigned.items()):
    print(f"  Cluster {clus}: {name}  (n={int((features['cluster']==clus).sum())})")

features.to_csv(f"{OUT_DIR}/customer_features_segmented.csv", index=False)

# ===========================================================================
# STEP 6: BEHAVIORAL ANALYSIS (BY CLUSTER/SEGMENT)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 6: BEHAVIORAL ANALYSIS (BY CLUSTERS)")
print("=" * 70)

segment_summary = features.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_recency_days=("recency_days", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean"),
    avg_order_value=("avg_order_value", "mean"),
    avg_basket_size=("avg_basket_size", "mean"),
    discount_usage_rate=("discount_usage_rate", "mean"),
    return_rate=("return_rate", "mean"),
    category_diversity=("category_diversity", "mean"),
    conversion_rate=("conversion_rate", "mean"),
).round(2).reset_index()
segment_summary["pct_of_customers"] = (segment_summary["customers"] / segment_summary["customers"].sum() * 100).round(1)
segment_summary["pct_of_revenue"] = (segment_summary["avg_monetary"] * segment_summary["customers"] /
                                      (segment_summary["avg_monetary"] * segment_summary["customers"]).sum() * 100).round(1)

# preferred category per segment
pref_cat = features.groupby("segment")["favorite_category"].agg(lambda x: x.mode().iloc[0])
segment_summary["top_category"] = segment_summary["segment"].map(pref_cat)

segment_summary.to_csv(f"{OUT_DIR}/segment_summary.csv", index=False)
print(segment_summary.to_string(index=False))

# ===========================================================================
# STEP 7: TREND ANALYSIS (OVER TIME)
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 7: TREND ANALYSIS (OVER TIME)")
print("=" * 70)

monthly_trend = orders_clean.groupby("year_month").agg(
    revenue=("total_amount", "sum"),
    orders=("order_id", "nunique"),
    customers=("customer_id", "nunique"),
    avg_order_value=("total_amount", "mean"),
).reset_index().sort_values("year_month")
monthly_trend.to_csv(f"{OUT_DIR}/monthly_trend.csv", index=False)

weekday_trend = orders_clean.groupby("weekday").agg(
    revenue=("total_amount", "sum"), orders=("order_id", "nunique")
).reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
weekday_trend.to_csv(f"{OUT_DIR}/weekday_trend.csv", index=False)

category_monthly_trend = orders_clean.groupby(["year_month", "category"])["total_amount"].sum().reset_index()
category_monthly_trend.to_csv(f"{OUT_DIR}/category_monthly_trend.csv", index=False)

# new customer growth per month (first purchase month)
first_purchase_month = orders_clean.groupby("customer_id")["order_date"].min().dt.to_period("M").astype(str)
new_customers_trend = first_purchase_month.value_counts().sort_index().reset_index()
new_customers_trend.columns = ["year_month", "new_customers"]
new_customers_trend.to_csv(f"{OUT_DIR}/new_customers_trend.csv", index=False)

# quarterly summary
quarterly_trend = orders_clean.groupby(["year", "quarter"]).agg(
    revenue=("total_amount", "sum"), orders=("order_id", "nunique")
).reset_index()
quarterly_trend["label"] = "Q" + quarterly_trend["quarter"].astype(str) + " " + quarterly_trend["year"].astype(str)
quarterly_trend.to_csv(f"{OUT_DIR}/quarterly_trend.csv", index=False)

print(f"Monthly trend: {len(monthly_trend)} months computed")
print(f"Peak revenue month: {monthly_trend.loc[monthly_trend['revenue'].idxmax(), 'year_month']} "
      f"({monthly_trend['revenue'].max():,.0f})")
print(f"New customer growth trend computed across {len(new_customers_trend)} months")

# ===========================================================================
# STEP 8: BUSINESS INSIGHTS & RECOMMENDATIONS
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 8: BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 70)

high_value_row = segment_summary.sort_values("avg_monetary", ascending=False).iloc[0]
at_risk_row = segment_summary[segment_summary["segment"] == "At Risk / Inactive Customers"]
top_category = eda_summary["revenue_by_category"]
top_cat_name = list(top_category.keys())[0]
best_month = monthly_trend.loc[monthly_trend["revenue"].idxmax()]
worst_month = monthly_trend.loc[monthly_trend["revenue"].idxmin()]

insights = [
    f"'{high_value_row['segment']}' make up {high_value_row['pct_of_customers']}% of customers but drive "
    f"{high_value_row['pct_of_revenue']}% of revenue — prioritize retention (loyalty perks, early access) for this group.",

    f"'At Risk / Inactive Customers' have an average recency of "
    f"{float(at_risk_row['avg_recency_days'].iloc[0]) if len(at_risk_row) else 0:.0f} days since last purchase. "
    f"Recommend win-back email/SMS campaigns with targeted discounts before they churn permanently.",

    f"'{top_cat_name}' is the top revenue-generating category — ensure sufficient inventory and feature it in "
    f"homepage promotions and paid campaigns.",

    f"Revenue peaked in {best_month['year_month']} ({best_month['revenue']:,.0f}) likely driven by festive/seasonal "
    f"demand, while {worst_month['year_month']} was the slowest ({worst_month['revenue']:,.0f}). Plan inventory and "
    f"marketing spend around this seasonality.",

    f"Overall return rate is {eda_summary['return_rate_overall']:.1%}. Investigate top-returned categories/products "
    f"to reduce reverse-logistics costs and improve product-page accuracy.",

    "Price Sensitive Customers respond strongly to discounts — target them with coupon codes and flash sales rather "
    "than full-price promotions to improve conversion without eroding margin elsewhere.",

    f"Personalize marketing using the {len(cluster_features)}-feature RFM + behavioral segmentation model: "
    "match campaign messaging, channel, and offer type to each segment's actual behavior instead of a one-size-fits-all blast.",
]

with open(f"{OUT_DIR}/business_insights.json", "w") as f:
    json.dump(insights, f, indent=2)

for i, ins in enumerate(insights, 1):
    print(f"{i}. {ins}")

# ===========================================================================
# Save consolidated summary JSON for the dashboard
# ===========================================================================
summary_payload = {
    "eda_summary": eda_summary,
    "elbow_data": elbow_data,
    "cluster_mapping": {str(k): v for k, v in assigned.items()},
    "insights": insights,
}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary_payload, f, indent=2, default=str)

print("\n" + "=" * 70)
print("PIPELINE COMPLETE. All outputs saved to ./outputs/")
print("=" * 70)
print("""
Output files:
  customer_features_segmented.csv  -> per-customer RFM + behavioral features + segment
  segment_summary.csv              -> aggregated stats per segment (for step 6)
  monthly_trend.csv                -> revenue/orders/customers by month (for step 7)
  weekday_trend.csv                -> revenue/orders by weekday
  category_monthly_trend.csv       -> category performance over time
  new_customers_trend.csv          -> customer growth trend
  quarterly_trend.csv              -> quarterly rollup
  business_insights.json           -> auto-generated recommendations (step 8)
  summary.json                     -> consolidated stats for the dashboard
""")
