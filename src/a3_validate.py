"""Part A, task 6: validation of reference checkpoints (CSV and SQLite)."""
import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
con = sqlite3.connect("data/instamart.db")

comp = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]
cities = c["city"].value_counts().to_dict()

checks = [
    ("12,309 order rows", len(o) == 12309),
    ("order_id unique", o["order_id"].is_unique),
    ("11,110 completed", len(comp) == 11110),
    ("1,199 cancelled", len(canc) == 1199),
    ("no pending rows", int((o["status"] == "pending").sum()) == 0),
    ("0 missing order_amount", int(o["order_amount"].isna().sum()) == 0),
    ("0 missing delivery (completed)", int(comp["delivery_minutes"].isna().sum()) == 0),
    ("1,199 missing delivery (cancelled)", int(canc["delivery_minutes"].isna().sum()) == 1199),
    ("completed revenue 2,998,668.80", round(comp["order_amount"].sum(), 2) == 2998668.80),
    ("six cities", len(cities) == 6),
    ("city counts", cities == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts", c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("0 missing ages", int(c["age"].isna().sum()) == 0),
    ("ages within 18-60", bool(c["age"].between(18, 60).all())),
    ("DB customers rows", pd.read_sql_query("select count(*) n from customers", con)["n"][0] == 1200),
    ("DB orders rows", pd.read_sql_query("select count(*) n from orders", con)["n"][0] == 12309),
    ("DB products rows", pd.read_sql_query("select count(*) n from products", con)["n"][0] == 40),
]
ok = True
for name, passed in checks:
    print(("PASS " if passed else "FAIL ") + name)
    ok &= bool(passed)
assert ok, "some checkpoints failed"
print("ALL CHECKPOINTS PASSED")
