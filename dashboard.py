"""
dashboard.py
------------
Step 9: DASHBOARD & VISUALIZATION
Interactive Streamlit dashboard for the E-Commerce Analytics Flow:
  Trend Analysis + Customer Segmentation + Behavioral Analysis

Run:
    streamlit run dashboard.py

Prerequisite: run generate_dataset.py then analytics_pipeline.py first
(or just run this file — it will auto-run them if outputs are missing).
"""

import os
import subprocess
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = "data"
OUT_DIR = "outputs"

# Auto-generate data/run pipeline if this is a fresh clone
if not os.path.exists(f"{DATA_DIR}/orders.csv"):
    with st.spinner("First run detected: generating synthetic dataset..."):
        subprocess.run(["python", "generate_dataset.py"], check=True)
if not os.path.exists(f"{OUT_DIR}/summary.json"):
    with st.spinner("Running analytics pipeline (cleaning, RFM, clustering, trends)..."):
        subprocess.run(["python", "analytics_pipeline.py"], check=True)


# ---------------------------------------------------------------------------
# LOAD DATA (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_all():
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv", parse_dates=["order_date"])
    features = pd.read_csv(f"{OUT_DIR}/customer_features_segmented.csv")
    segment_summary = pd.read_csv(f"{OUT_DIR}/segment_summary.csv")
    monthly_trend = pd.read_csv(f"{OUT_DIR}/monthly_trend.csv")
    weekday_trend = pd.read_csv(f"{OUT_DIR}/weekday_trend.csv")
    category_monthly_trend = pd.read_csv(f"{OUT_DIR}/category_monthly_trend.csv")
    new_customers_trend = pd.read_csv(f"{OUT_DIR}/new_customers_trend.csv")
    quarterly_trend = pd.read_csv(f"{OUT_DIR}/quarterly_trend.csv")
    with open(f"{OUT_DIR}/summary.json") as f:
        summary = json.load(f)
    return (orders, features, segment_summary, monthly_trend, weekday_trend,
            category_monthly_trend, new_customers_trend, quarterly_trend, summary)


(orders, features, segment_summary, monthly_trend, weekday_trend,
 category_monthly_trend, new_customers_trend, quarterly_trend, summary) = load_all()

orders["is_returned"] = orders["order_status"].eq("Returned")
orders_valid = orders[orders["order_status"] != "Cancelled"].copy()

SEGMENT_COLORS = {
    "High Value Customers": "#2ca02c",
    "New Customers": "#1f77b4",
    "Price Sensitive Customers": "#ff7f0e",
    "Occasional Buyers": "#9467bd",
    "At Risk / Inactive Customers": "#d62728",
}

# ---------------------------------------------------------------------------
# DATA SETUP
# ---------------------------------------------------------------------------
orders_f = orders_valid.copy()
features_f = features.copy()
sel_categories = sorted(orders_f["category"].unique())
sel_segments = sorted(features_f["segment"].unique())

# ---------------------------------------------------------------------------
# HEADER + KPIs
# ---------------------------------------------------------------------------

def add_recommendation(text):
    st.info(f"💡 Recommendation: {text}")


def summarize_growth(values, label):
    if len(values) < 2:
        return f"{label} is stable right now; keep monitoring and protect current conversion rates."

    start = float(values.iloc[0])
    end = float(values.iloc[-1])

    if end > start:
        change_pct = ((end - start) / start * 100) if start else 0
        return f"{label} increased by {change_pct:.1f}% from {start:,.0f} to {end:,.0f}. Lean into the best-performing channels and protect that momentum."
    if end < start:
        change_pct = ((start - end) / start * 100) if start else 0
        return f"{label} fell by {change_pct:.1f}% from {start:,.0f} to {end:,.0f}. Review promotions, acquisition, and product availability to recover performance."
    return f"{label} stayed flat. Keep the current plan steady while tracking seasonality closely."


st.title("E-Commerce Analytics Dashboard")
st.caption("Trend Analysis · Customer Segmentation · Behavioral Analysis")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"₹{orders_f['total_amount'].sum():,.0f}")
k2.metric("Total Orders", f"{orders_f['order_id'].nunique():,}")
k3.metric("Active Customers", f"{orders_f['customer_id'].nunique():,}")
k4.metric("Avg Order Value", f"₹{orders_f['total_amount'].mean():,.0f}")
k5.metric("Return Rate", f"{orders_f['is_returned'].mean():.1%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [" Trend Analysis", "Customer Segmentation", " Behavioral Analysis", " Conclusion"]
)

# =========================== TAB 1: TREND ANALYSIS =========================
with tab1:
    st.subheader("Monthly Revenue & Order Trend")
    mt = orders_f.groupby(orders_f["order_date"].dt.to_period("M").astype(str)).agg(
        revenue=("total_amount", "sum"), orders=("order_id", "nunique"),
        customers=("customer_id", "nunique")
    ).reset_index().rename(columns={"order_date": "year_month"})
    mt.columns = ["year_month", "revenue", "orders", "customers"]
    mt = mt.sort_values("year_month")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mt["year_month"], y=mt["revenue"], name="Revenue",
                              mode="lines+markers", line=dict(color="#2ca02c", width=3)))
    fig.update_layout(height=400, xaxis_title="Month", yaxis_title="Revenue (₹)",
                       hovermode="x unified", margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    add_recommendation(summarize_growth(mt["revenue"], "Revenue"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Orders per Month")
        fig2 = px.bar(mt, x="year_month", y="orders", color_discrete_sequence=["#1f77b4"])
        fig2.update_layout(height=350, margin=dict(t=20))
        st.plotly_chart(fig2, width='stretch')
        add_recommendation(summarize_growth(mt["orders"], "Order volume"))
    with c2:
        st.subheader("Revenue by Day of Week")
        wt = orders_f.groupby(orders_f["order_date"].dt.day_name())["total_amount"].sum()
        wt = wt.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        fig3 = px.bar(x=wt.index, y=wt.values, color_discrete_sequence=["#ff7f0e"])
        fig3.update_layout(height=350, xaxis_title="", yaxis_title="Revenue (₹)", margin=dict(t=20))
        st.plotly_chart(fig3, width='stretch')
        peak_day = wt.idxmax()
        peak_value = wt.max()
        add_recommendation(f"{peak_day} generated the strongest revenue ({peak_value:,.0f} ₹). Focus promotional campaigns and support staffing around this peak day.")

    st.subheader("Category / Product Trend Over Time")
    cat_trend = orders_f.groupby([orders_f["order_date"].dt.to_period("M").astype(str), "category"])["total_amount"] \
        .sum().reset_index()
    cat_trend.columns = ["year_month", "category", "revenue"]
    fig4 = px.area(cat_trend, x="year_month", y="revenue", color="category")
    fig4.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(fig4, width='stretch')
    top_category = cat_trend.groupby("category")["revenue"].sum().idxmax()
    add_recommendation(f"{top_category} is the strongest category. Increase inventory and cross-sell placements around this product family.")

    st.subheader("Customer Growth Trend (New Customers per Month)")
    fig5 = px.line(new_customers_trend, x="year_month", y="new_customers", markers=True,
                    color_discrete_sequence=["#9467bd"])
    fig5.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig5, width='stretch')
    add_recommendation(summarize_growth(new_customers_trend["new_customers"], "New customer acquisition"))

    st.subheader("Seasonal / Festival Impact (Quarterly)")
    fig6 = px.bar(quarterly_trend, x="label", y="revenue", color="revenue",
                   color_continuous_scale="Greens")
    fig6.update_layout(height=350, xaxis_title="Quarter", margin=dict(t=20))
    st.plotly_chart(fig6, width='stretch')
    peak_quarter = quarterly_trend.loc[quarterly_trend["revenue"].idxmax(), "label"]
    add_recommendation(f"{peak_quarter} contributes the most revenue. Use it as the anchor for seasonal campaigns and stock planning.")

# ======================= TAB 2: CUSTOMER SEGMENTATION =======================
with tab2:
    st.subheader("Customer Segments (K-Means Clustering on RFM + Behavioral Features)")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        seg_counts = features_f["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "customers"]
        fig = px.pie(seg_counts, names="segment", values="customers", hole=0.45,
                     color="segment", color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(height=420, margin=dict(t=20))
        st.plotly_chart(fig, width='stretch')
        largest_segment = seg_counts.loc[seg_counts["customers"].idxmax(), "segment"]
        add_recommendation(f"{largest_segment} is the largest segment, so protect this base with loyalty offers and proactive retention.")
    with c2:
        fig = px.scatter(features_f, x="recency_days", y="monetary", size="frequency",
                          color="segment", color_discrete_map=SEGMENT_COLORS,
                          hover_data=["customer_name", "frequency", "avg_order_value"],
                          labels={"recency_days": "Recency (days since last order)",
                                  "monetary": "Total Spend (₹)"})
        fig.update_layout(height=420, margin=dict(t=20))
        st.plotly_chart(fig, width='stretch')
        high_value_segment = features_f.groupby("segment")["monetary"].mean().idxmax()
        add_recommendation(f"{high_value_segment} shows the strongest spend profile. Offer premium benefits and upsells to increase lifetime value.")

    st.subheader("Segment Profile Summary")
    display_cols = ["segment", "customers", "pct_of_customers", "pct_of_revenue",
                     "avg_recency_days", "avg_frequency", "avg_monetary", "avg_order_value", "top_category"]
    st.dataframe(
        segment_summary[segment_summary["segment"].isin(sel_segments)][display_cols]
        .sort_values("avg_monetary", ascending=False)
        .rename(columns={
            "pct_of_customers": "% of Customers", "pct_of_revenue": "% of Revenue",
            "avg_recency_days": "Avg Recency (days)", "avg_frequency": "Avg Orders",
            "avg_monetary": "Avg Lifetime Spend (₹)", "avg_order_value": "Avg Order Value (₹)",
            "top_category": "Top Category"
        }),
        width='stretch', hide_index=True
    )
    add_recommendation("Use this table to allocate budget: reward high-value customers, revive inactive ones, and test new offers on fast-growing segments.")

    st.subheader("RFM Distribution by Segment")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.box(features_f, x="segment", y="recency_days", color="segment",
                      color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(height=350, showlegend=False, xaxis_title="", margin=dict(t=20))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width='stretch')
        recency_leader = features_f.groupby("segment")["recency_days"].mean().idxmin()
        add_recommendation(f"{recency_leader} is the most recently active segment. Keep them engaged with loyalty rewards and early-access offers.")
    with c2:
        fig = px.box(features_f, x="segment", y="frequency", color="segment",
                      color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(height=350, showlegend=False, xaxis_title="", margin=dict(t=20))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width='stretch')
        frequency_leader = features_f.groupby("segment")["frequency"].mean().idxmax()
        add_recommendation(f"{frequency_leader} buys most often. Use bundle offers and cross-sells to lift basket size further.")
    with c3:
        fig = px.box(features_f, x="segment", y="monetary", color="segment",
                      color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(height=350, showlegend=False, xaxis_title="", margin=dict(t=20))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width='stretch')
        monetary_leader = features_f.groupby("segment")["monetary"].mean().idxmax()
        add_recommendation(f"{monetary_leader} delivers the highest spend. Give this segment VIP treatment and retention offers to protect revenue.")

    with st.expander("How segments were determined (Elbow Method / Silhouette Score)"):
        elbow = summary["elbow_data"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elbow["k"], y=elbow["inertia"], name="Inertia (Elbow)",
                                  yaxis="y1", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=elbow["k"], y=elbow["silhouette"], name="Silhouette Score",
                                  yaxis="y2", mode="lines+markers"))
        fig.update_layout(
            height=350,
            yaxis=dict(title="Inertia"),
            yaxis2=dict(title="Silhouette Score", overlaying="y", side="right"),
            xaxis=dict(title="Number of Clusters (k)"),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, width='stretch')
        add_recommendation("The clustering appears stable at five segments, so keep this model and review it quarterly as customer behavior shifts.")
        st.caption("Final model uses k=5 clusters, mapped to business-friendly segment names "
                   "based on each cluster's average recency, frequency, and monetary value.")

    st.subheader("Explore Individual Customers")
    st.dataframe(
        features_f[["customer_id", "customer_name", "segment", "recency_days", "frequency",
                    "monetary", "avg_order_value", "favorite_category", "city"]]
        .sort_values("monetary", ascending=False).head(200),
        width='stretch', hide_index=True
    )

# ======================= TAB 3: BEHAVIORAL ANALYSIS =========================
with tab3:
    st.subheader("Purchase Behavior by Segment")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(segment_summary[segment_summary["segment"].isin(sel_segments)],
                      x="segment", y="avg_order_value", color="segment",
                      color_discrete_map=SEGMENT_COLORS, title="Avg Order Value by Segment")
        fig.update_layout(height=380, showlegend=False, xaxis_title="", margin=dict(t=40))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width='stretch')
        highest_aov_segment = segment_summary.loc[segment_summary["avg_order_value"].idxmax(), "segment"]
        add_recommendation(f"{highest_aov_segment} has the highest average order value. Encourage bundle purchases and premium product discovery for this customer group.")
    with c2:
        fig = px.bar(segment_summary[segment_summary["segment"].isin(sel_segments)],
                      x="segment", y="discount_usage_rate", color="segment",
                      color_discrete_map=SEGMENT_COLORS, title="Discount Usage Rate by Segment")
        fig.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_tickformat=".0%",
                           margin=dict(t=40))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width='stretch')
        highest_discount_segment = segment_summary.loc[segment_summary["discount_usage_rate"].idxmax(), "segment"]
        add_recommendation(f"{highest_discount_segment} uses discounts most heavily. Review offer depth and use value-based messaging rather than broad discounting.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(segment_summary[segment_summary["segment"].isin(sel_segments)],
                      x="segment", y="return_rate", color="segment",
                      color_discrete_map=SEGMENT_COLORS, title="Return Rate by Segment")
        fig.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_tickformat=".0%",
                           margin=dict(t=40))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width='stretch')
        highest_return_segment = segment_summary.loc[segment_summary["return_rate"].idxmax(), "segment"]
        add_recommendation(f"{highest_return_segment} has the highest return rate. Improve product descriptions, size guidance, and support to reduce post-purchase friction.")
    with c2:
        fig = px.bar(segment_summary[segment_summary["segment"].isin(sel_segments)],
                      x="segment", y="category_diversity", color="segment",
                      color_discrete_map=SEGMENT_COLORS, title="Category Diversity by Segment")
        fig.update_layout(height=380, showlegend=False, xaxis_title="", margin=dict(t=40))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, width='stretch')
        most_diverse_segment = segment_summary.loc[segment_summary["category_diversity"].idxmax(), "segment"]
        add_recommendation(f"{most_diverse_segment} explores the widest range of categories. Cross-sell complementary products to increase basket size.")

    st.subheader("Preferred Category by Segment")
    pref = features_f.groupby(["segment", "favorite_category"]).size().reset_index(name="customers")
    fig = px.bar(pref, x="segment", y="customers", color="favorite_category", barmode="stack")
    fig.update_layout(height=420, xaxis_title="", margin=dict(t=20))
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, width='stretch')
    top_pref = pref.loc[pref["customers"].idxmax()]
    add_recommendation(f"{top_pref['segment']} prefers {top_pref['favorite_category']}. Tailor landing pages and offers around this category to boost conversion.")

    st.subheader("Time of Purchase Pattern (Weekday) — Filtered Customers")
    seg_customer_ids = features_f["customer_id"].unique()
    seg_orders = orders_f[orders_f["customer_id"].isin(seg_customer_ids)]
    weekday_pattern = seg_orders["order_date"].dt.day_name().value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig = px.line_polar(r=weekday_pattern.values, theta=weekday_pattern.index, line_close=True)
    fig.update_traces(fill="toself")
    fig.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    peak_purchase_day = weekday_pattern.idxmax()
    add_recommendation(f"{peak_purchase_day} is the busiest purchase day. Time campaigns and push notifications around this cadence.")

# ========================= TAB 4: CONCLUSION ==========================
with tab4:
    st.subheader("Conclusion & Recommended Actions")
    for i, insight in enumerate(summary.get("insights", []), 1):
        st.success(f"**{i}.** {insight}")

    st.subheader("Segment Comparison Radar")
    radar_metrics = ["avg_recency_days", "avg_frequency", "avg_monetary", "avg_order_value", "category_diversity"]
    radar_df = segment_summary[segment_summary["segment"].isin(sel_segments)].copy()
    norm = radar_df[radar_metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
    fig = go.Figure()
    for i, row in radar_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=norm.loc[i].values.tolist() + [norm.loc[i].values[0]],
            theta=radar_metrics + [radar_metrics[0]],
            fill="toself", name=row["segment"],
            line=dict(color=SEGMENT_COLORS.get(row["segment"]))
        ))
    fig.update_layout(height=500, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, width='stretch')
    add_recommendation("Prioritize retention for high-value segments while using win-back campaigns for inactive customers to improve profitability.")

    st.subheader("Revenue Contribution: Segment vs Customer Share")
    fig = px.scatter(segment_summary[segment_summary["segment"].isin(sel_segments)],
                      x="pct_of_customers", y="pct_of_revenue", size="avg_monetary",
                      color="segment", color_discrete_map=SEGMENT_COLORS, text="segment")
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dot", color="gray"))
    fig.update_traces(textposition="top center")
    fig.update_layout(height=450, xaxis_title="% of Customers", yaxis_title="% of Revenue", margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    add_recommendation("Invest more in the segments above the diagonal line because they generate revenue disproportionate to their size.")
    st.caption("Segments above the diagonal line generate more revenue share than their customer share — "
               "the classic 80/20 pattern to prioritize in retention strategy.")

st.markdown("---")
st.caption("E-Commerce Analytics Flow — Data Collection → Cleaning → Feature Engineering → EDA → "
           "Segmentation → Behavioral Analysis → Trend Analysis → Conclusion → Dashboard")
