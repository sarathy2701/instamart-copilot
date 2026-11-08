"""Part B, tasks 1-3: RFM table, K-means, segment profile/names, PCA, outputs/customer_segments.csv."""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
SNAP = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = o[o["status"] == "completed"]

# ---- task 1: RFM
g = comp.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (SNAP - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum().round(2),
}).reset_index()
print("RFM customers:", len(rfm))
print("recency median %d min %d max %d" % (rfm.recency.median(), rfm.recency.min(), rfm.recency.max()))
print("frequency mean %.2f max %d" % (rfm.frequency.mean(), rfm.frequency.max()))
print("monetary mean %.2f max %.2f" % (rfm.monetary.mean(), rfm.monetary.max()))

# ---- task 2: K-means sweep
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_)))
tab = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
print("\nK-means sweep:")
print(tab.round(3).to_string(index=False))

# ---- task 3: profile, names, PCA
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean"))
print("\nsegment profile (K=%d):" % FINAL_K)
print(prof.round(2).to_string())

# Cluster ids are arbitrary labels with no order, so names are assigned from the profile, never from the id.
names, left = {}, set(prof.index)
premium_id = prof["monetary"].idxmax(); names[premium_id] = "Premium Big Spenders"; left.discard(premium_id)
loyal_id = prof.loc[list(left), "frequency"].idxmax(); names[loyal_id] = "Loyal Regulars"; left.discard(loyal_id)
lapsing_id = prof.loc[list(left), "recency"].idxmax(); names[lapsing_id] = "Lapsing / Dormant"; left.discard(lapsing_id)
for s in left:
    names[s] = "Occasional Buyers"
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
actions = {"Premium Big Spenders": "Early access to electronics drops and a dedicated premium support desk.",
           "Loyal Regulars": "Subscription / bundle offers on groceries and dairy to lock in the habit.",
           "Lapsing / Dormant": "Win-back coupon plus a reminder push before they churn for good.",
           "Occasional Buyers": "Cross-sell nudges and a free-delivery threshold offer to raise order frequency."}
print("\nnames and actions:")
for s in prof.index:
    print(s, names[s], "|", actions[names[s]])
print("premium segment id:", premium_id)

pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print("\nPCA explained_variance_ratio_: %.3f %.3f total %.3f" % (r[0], r[1], r.sum()))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)
print("wrote outputs/customer_segments.csv rows:", len(rfm), "| premium rows:", int(rfm.is_premium.sum()))
