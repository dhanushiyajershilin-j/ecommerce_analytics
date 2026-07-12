"""
generate_dataset.py
--------------------
Generates a realistic synthetic e-commerce dataset for the analytics pipeline:
    1. customers.csv         -> customer master data
    2. products.csv          -> product catalog
    3. orders.csv             -> order / transaction line items (the core fact table)
    4. website_sessions.csv  -> website behavior / clickstream summary

Run:  python generate_dataset.py
Output written to ./data/
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
N_CUSTOMERS = 3000
N_PRODUCTS = 250
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)
N_ORDERS = 42000          # order line items
CITIES = ["Chennai", "Coimbatore", "Bengaluru", "Mumbai", "Delhi", "Hyderabad",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Tiruppur"]
STATES = {
    "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu", "Tiruppur": "Tamil Nadu",
    "Bengaluru": "Karnataka", "Mumbai": "Maharashtra", "Pune": "Maharashtra",
    "Delhi": "Delhi", "Hyderabad": "Telangana", "Kolkata": "West Bengal",
    "Ahmedabad": "Gujarat", "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh"
}
CATEGORIES = {
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Cameras", "Accessories"],
    "Fashion": ["Men's Wear", "Women's Wear", "Footwear", "Watches", "Bags"],
    "Home & Kitchen": ["Cookware", "Furniture", "Decor", "Appliances", "Storage"],
    "Beauty & Personal Care": ["Skincare", "Haircare", "Makeup", "Fragrance"],
    "Sports & Fitness": ["Gym Equipment", "Sportswear", "Outdoor Gear"],
    "Books & Stationery": ["Fiction", "Non-Fiction", "Office Supplies"],
    "Grocery": ["Snacks", "Beverages", "Staples", "Organic"]
}
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery", "Wallet"]
DEVICES = ["Mobile", "Desktop", "Tablet"]

# Festival / seasonality boost windows (month, day-range) -> demand multiplier
SEASONAL_BOOSTS = [
    ((1, 1), (1, 15), 1.3),     # New Year sale
    ((3, 1), (3, 20), 1.2),     # Spring sale
    ((8, 1), (8, 31), 1.4),     # Independence day / mid-year sale
    ((10, 1), (10, 31), 1.8),   # Diwali / festive season - big boost
    ((11, 20), (11, 30), 1.6),  # Black Friday
    ((12, 15), (12, 31), 1.5),  # Christmas / year-end sale
]


def seasonal_multiplier(date):
    mult = 1.0
    for (m1, d1), (m2, d2), boost in SEASONAL_BOOSTS:
        start = datetime(date.year, m1, d1)
        end = datetime(date.year, m2, d2)
        if start <= date <= end:
            mult = max(mult, boost)
    # gentle year-over-year growth trend
    days_from_start = (date - START_DATE).days
    growth = 1 + (days_from_start / (END_DATE - START_DATE).days) * 0.35
    return mult * growth


def random_dates(start, end, n):
    delta_days = (end - start).days
    # weight recent dates & seasonal peaks slightly higher for realism
    offsets = np.random.randint(0, delta_days, size=n)
    dates = [start + timedelta(days=int(o)) for o in offsets]
    return dates


# ---------------------------------------------------------------------------
# 1. CUSTOMERS
# ---------------------------------------------------------------------------
print("Generating customers...")
first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Krishna",
               "Ishaan", "Rohan", "Ananya", "Diya", "Isha", "Kavya", "Meera", "Priya",
               "Riya", "Saanvi", "Tara", "Zoya", "Karthik", "Naveen", "Deepak", "Suresh",
               "Lakshmi", "Divya", "Pooja", "Neha", "Rahul", "Amit"]
last_names = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Menon", "Gupta", "Patel",
              "Kumar", "Singh", "Rao", "Pillai", "Chatterjee", "Mukherjee", "Desai", "Joshi"]

customer_ids = [f"CUST{100000+i}" for i in range(N_CUSTOMERS)]
signup_dates = random_dates(START_DATE - timedelta(days=365), END_DATE, N_CUSTOMERS)

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "customer_name": [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(N_CUSTOMERS)],
    "gender": np.random.choice(["Male", "Female", "Other"], N_CUSTOMERS, p=[0.48, 0.49, 0.03]),
    "age": np.random.randint(18, 65, N_CUSTOMERS),
    "city": np.random.choice(CITIES, N_CUSTOMERS),
    "signup_date": signup_dates,
})
customers["state"] = customers["city"].map(STATES)
customers["signup_date"] = pd.to_datetime(customers["signup_date"]).dt.strftime("%Y-%m-%d")

# introduce some messiness for the cleaning step to fix
messy_idx = np.random.choice(customers.index, size=int(0.02 * N_CUSTOMERS), replace=False)
customers.loc[messy_idx, "age"] = np.nan
dup_rows = customers.sample(15, random_state=1)
customers = pd.concat([customers, dup_rows], ignore_index=True)

customers.to_csv(f"{OUT_DIR}/customers.csv", index=False)
print(f"  -> {len(customers)} customer records")

# ---------------------------------------------------------------------------
# 2. PRODUCTS
# ---------------------------------------------------------------------------
print("Generating products...")
prod_rows = []
pid = 1
for cat, subs in CATEGORIES.items():
    n_this_cat = N_PRODUCTS // len(CATEGORIES)
    for _ in range(n_this_cat):
        sub = np.random.choice(subs)
        base_price = {
            "Electronics": np.random.uniform(800, 60000),
            "Fashion": np.random.uniform(300, 5000),
            "Home & Kitchen": np.random.uniform(200, 15000),
            "Beauty & Personal Care": np.random.uniform(100, 3000),
            "Sports & Fitness": np.random.uniform(250, 12000),
            "Books & Stationery": np.random.uniform(50, 1200),
            "Grocery": np.random.uniform(30, 900),
        }[cat]
        prod_rows.append({
            "product_id": f"P{1000+pid}",
            "category": cat,
            "sub_category": sub,
            "product_name": f"{sub} Item {pid}",
            "unit_price": round(base_price, 2)
        })
        pid += 1
products = pd.DataFrame(prod_rows)
products.to_csv(f"{OUT_DIR}/products.csv", index=False)
print(f"  -> {len(products)} product records")

# ---------------------------------------------------------------------------
# 3. ORDERS (core transactional fact table)
# ---------------------------------------------------------------------------
print("Generating orders (this is the big one)...")

# give customers heterogeneous purchase propensity (some are big spenders / frequent buyers)
propensity = np.random.gamma(shape=2.0, scale=1.0, size=N_CUSTOMERS)
propensity = propensity / propensity.sum()
cust_choice_ids = customers["customer_id"].values[:N_CUSTOMERS]

order_dates_raw = random_dates(START_DATE, END_DATE, N_ORDERS)
# apply seasonal reweighting by resampling extra dates and keeping ones that pass a probability filter
final_dates = []
pool = random_dates(START_DATE, END_DATE, N_ORDERS * 2)
for d in pool:
    if len(final_dates) >= N_ORDERS:
        break
    p = seasonal_multiplier(d) / 2.2  # normalize roughly into 0-1
    if np.random.rand() < min(p, 1.0):
        final_dates.append(d)
while len(final_dates) < N_ORDERS:
    final_dates.append(np.random.choice(pool))
order_dates = final_dates[:N_ORDERS]

chosen_customers = np.random.choice(cust_choice_ids, size=N_ORDERS, p=propensity)
chosen_products = products.sample(N_ORDERS, replace=True).reset_index(drop=True)

orders = pd.DataFrame({
    "order_id": [f"ORD{500000+i}" for i in range(N_ORDERS)],
    "customer_id": chosen_customers,
    "order_date": pd.to_datetime(order_dates),
    "product_id": chosen_products["product_id"],
    "category": chosen_products["category"],
    "sub_category": chosen_products["sub_category"],
    "unit_price": chosen_products["unit_price"],
})

orders["quantity"] = np.random.choice([1, 1, 1, 2, 2, 3, 4], N_ORDERS)
orders["discount_pct"] = np.random.choice([0, 0, 0, 5, 10, 15, 20, 25, 30], N_ORDERS,
                                           p=[0.35, 0.1, 0.1, 0.12, 0.12, 0.08, 0.06, 0.04, 0.03])
orders["gross_amount"] = (orders["unit_price"] * orders["quantity"]).round(2)
orders["discount_amount"] = (orders["gross_amount"] * orders["discount_pct"] / 100).round(2)
orders["total_amount"] = (orders["gross_amount"] - orders["discount_amount"]).round(2)
orders["payment_method"] = np.random.choice(PAYMENT_METHODS, N_ORDERS,
                                             p=[0.22, 0.18, 0.32, 0.1, 0.13, 0.05])
orders["delivery_days"] = np.random.randint(1, 10, N_ORDERS)

# order status: mostly delivered, some returned/cancelled
orders["order_status"] = np.random.choice(
    ["Delivered", "Returned", "Cancelled"], N_ORDERS, p=[0.88, 0.08, 0.04]
)

# a bit of missing / messy data to be cleaned later
messy_order_idx = np.random.choice(orders.index, size=int(0.015 * N_ORDERS), replace=False)
orders.loc[messy_order_idx, "delivery_days"] = np.nan
dup_orders = orders.sample(80, random_state=2)
orders = pd.concat([orders, dup_orders], ignore_index=True)

orders = orders.sort_values("order_date").reset_index(drop=True)
orders.to_csv(f"{OUT_DIR}/orders.csv", index=False)
print(f"  -> {len(orders)} order line items")

# ---------------------------------------------------------------------------
# 4. WEBSITE BEHAVIOR (sessions)
# ---------------------------------------------------------------------------
print("Generating website sessions...")
N_SESSIONS = 60000
sess_customers = np.random.choice(cust_choice_ids, size=N_SESSIONS, p=propensity)
sess_dates = random_dates(START_DATE, END_DATE, N_SESSIONS)

sessions = pd.DataFrame({
    "session_id": [f"SESS{900000+i}" for i in range(N_SESSIONS)],
    "customer_id": sess_customers,
    "visit_date": pd.to_datetime(sess_dates),
    "device": np.random.choice(DEVICES, N_SESSIONS, p=[0.62, 0.30, 0.08]),
    "pages_viewed": np.random.poisson(4, N_SESSIONS) + 1,
    "session_duration_min": np.round(np.random.exponential(4, N_SESSIONS) + 0.5, 2),
})
# a session "converts" (leads to purchase) with some probability influenced by pages viewed
conv_prob = 0.08 + 0.02 * np.log1p(sessions["pages_viewed"])
sessions["converted"] = (np.random.rand(N_SESSIONS) < conv_prob).astype(int)

sessions.to_csv(f"{OUT_DIR}/website_sessions.csv", index=False)
print(f"  -> {len(sessions)} session records")

print("\nAll raw data generated in ./data/")
print("Files: customers.csv, products.csv, orders.csv, website_sessions.csv")
