"""Part A, task 7: run the seven queries in sql/queries.sql with pd.read_sql_query and report results."""
import re
import sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}
con = sqlite3.connect("data/instamart.db")
r = {k: pd.read_sql_query(q, con) for k, q in queries.items()}

print("Q1 customers:", len(r["Q1"]))
q2 = r["Q2"]
print("Q2 rows:", len(q2), "| revenue sum: %.2f" % q2["revenue"].sum(), "| largest:", q2.iloc[0].to_dict())
print("Q3 products above average price:", len(r["Q3"]))
print("Q4 customers without completed order:", len(r["Q4"]))
print("Q5 customers above average:", len(r["Q5"]), "| average spend:", r["Q5"]["avg_total"].iloc[0])
q6 = r["Q6"]
print("Q6 top customer per city:")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()))
q7 = r["Q7"]
print("Q7 daily rows:", len(q7), "| last cumulative: %.2f" % q7["cumulative_revenue"].iloc[-1])
