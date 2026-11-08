"""Part A, tasks 3-5: clean customers and orders, write data/clean/*.csv and data/instamart.db."""
import os
import sqlite3
import pandas as pd

RAW, CLEAN, DB = "data/raw/", "data/clean/", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)

# ------------------------------------------------------------ customers
c = pd.read_csv(RAW + "customers.csv")
p = pd.read_csv(RAW + "products.csv")

c["city"] = c["city"].str.strip().str.title()
print("cities:", c["city"].value_counts().to_dict())

gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
print("genders:", c["gender"].value_counts().to_dict())

age_raw = pd.to_numeric(c["age"], errors="coerce")
bad_age = (age_raw < 10) | (age_raw > 100)
print("ages set to missing (outside 10-100):", int(bad_age.sum()))
c["age"] = age_raw.mask(bad_age)
n_missing_before_fill = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
print("missing ages before fill:", n_missing_before_fill, "| after:", int(c["age"].isna().sum()))
print("age range:", c["age"].min(), c["age"].max())
c["signup_date"] = pd.to_datetime(c["signup_date"])

# ------------------------------------------------------------ orders
o = pd.read_csv(RAW + "orders.csv")
n0 = len(o)
print("\nraw order rows:", n0)

o["order_ts"] = pd.to_datetime(o["order_ts"])

n = len(o); o = o.drop_duplicates(); print("rule 2: exact duplicates removed:", n - len(o))

n = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
print("rule 3: superseded (earlier pending) records removed:", n - len(o))
print("status counts:", o["status"].value_counts().to_dict())

amt = pd.to_numeric(o["order_amount"], errors="coerce")
n_bad = int(amt.isna().sum())
o["order_amount"] = amt
o = o.merge(p[["product_id", "category", "price_inr"]], on="product_id", how="left")
fill = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o["order_amount"] = o["order_amount"].fillna(fill)
print("rule 4: unparseable amounts rebuilt from price:", n_bad, "| remaining missing:", int(o["order_amount"].isna().sum()))

d = pd.to_numeric(o["delivery_minutes"], errors="coerce")
neg = int((d < 0).sum())
d = d.mask(d < 0)
o["delivery_minutes"] = d
comp = o["status"] == "completed"
miss_before = int((comp & d.isna()).sum())
med = o["delivery_minutes"].where(comp).groupby(o["category"]).transform("median")
o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(med[comp])
print("rule 5: negative delivery times set to missing:", neg,
      "| completed orders filled with category median:", miss_before,
      "| completed still missing:", int((comp & o["delivery_minutes"].isna()).sum()),
      "| cancelled left missing:", int(((o["status"] == "cancelled") & o["delivery_minutes"].isna()).sum()))

o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
print("final orders:", len(o), o["status"].value_counts().to_dict())
print("completed revenue INR %.2f" % o.loc[o["status"] == "completed", "order_amount"].sum())

# ------------------------------------------------------------ store
c_out, o_out = c.copy(), o.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out.to_csv(CLEAN + "customers.csv", index=False)
p.to_csv(CLEAN + "products.csv", index=False)
o_out.to_csv(CLEAN + "orders.csv", index=False)

if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
c_out.to_sql("customers", con, index=False)
p.to_sql("products", con, index=False)
o_out.to_sql("orders", con, index=False)
con.close()
print("wrote", CLEAN, "and", DB)
