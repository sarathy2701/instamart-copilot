"""Part A, task 8: pandas cross-check and exploratory analysis."""
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
comp = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(
    p[["product_id", "category"]], on="product_id")

# (a) pivot cross-check of Q2
pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("pivot grand total: %.2f" % pv.loc["All", "All"])
print(pv.round(2).to_string())

city = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count"))
print("\nper-city:")
print(city.round(2).to_string())

# (b) distribution
x = comp["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\nmean %.2f median %.2f skew %.2f" % (x.mean(), x.median(), x.skew()))
print("Q1 %.2f Q3 %.2f IQR %.2f lower fence %.2f upper fence %.2f" % (q1, q3, iqr, lo, hi))
print("outliers below:", int((x < lo).sum()), "above:", int((x > hi).sum()))
corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\ncorrelation matrix:")
print(corr.round(3).to_string())
import numpy as np
cols = list(corr.columns)
pairs = {(a, b): corr.loc[a, b] for i, a in enumerate(cols) for b in cols[i + 1:]}
best = max(pairs, key=lambda k: abs(pairs[k]))
print("largest |corr| pair:", best, round(pairs[best], 3))
