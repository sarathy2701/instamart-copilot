"""Part A, task 2: audit the raw exports (shape, dtypes, missing values, specific audit figures)."""
import pandas as pd

RAW = "data/raw/"
# Only truly empty cells count as missing. The text "N/A" is a planted bad value, not a blank,
# so default NA parsing is switched off (otherwise "N/A" would silently become NaN).
KW = dict(keep_default_na=False, na_values=[""])
cust = pd.read_csv(RAW + "customers.csv", **KW)
prod = pd.read_csv(RAW + "products.csv", **KW)
orders = pd.read_csv(RAW + "orders.csv", **KW)


def describe(name, df):
    print(f"\n=== {name}: shape={df.shape} ===")
    miss = df.isna().sum()
    t = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": miss, "missing_pct": (miss / len(df) * 100).round(2)})
    print(t.to_string())


for n, d in [("customers", cust), ("products", prod), ("orders", orders)]:
    describe(n, d)

amt = pd.to_numeric(orders["order_amount"], errors="coerce")
non_numeric = int((amt.isna() & orders["order_amount"].notna()).sum())
deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(cust["age"], errors="coerce")
gender_raw = cust["gender"].dropna().astype(str)

audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "rows with status pending": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": non_numeric,
    "negative delivery_minutes": int((deliv < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "customers: blank age": int(cust["age"].isna().sum()),
    "customers: age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "customers: distinct raw city spellings": int(cust["city"].nunique()),
    "customers: distinct non-blank raw gender spellings": int(gender_raw.nunique()),
    "customers: blank gender": int(cust["gender"].isna().sum()),
}
print("\n=== Audit figures ===")
for k, v in audit.items():
    print(f"{k}: {v}")
