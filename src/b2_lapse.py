"""Part B, tasks 4-8: lapse problem, leakage-free preprocessing, tuning, balance experiment, persistence."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
o = o.merge(p[["product_id", "category"]], on="product_id")

# ---------------- task 4: cohort, label, features (history strictly before the cutoff)
hist = o[o["order_ts"] < CUTOFF]
h_comp = hist[hist["status"] == "completed"]
cohort_ids = h_comp["customer_id"].unique()
sept = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < END) & (o["status"] == "completed")]
active_sept = set(sept["customer_id"])

g = h_comp.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0]),
})
n_cancel = hist[hist["status"] == "cancelled"].groupby("customer_id").size()
feat["cancel_rate"] = (n_cancel.reindex(feat.index).fillna(0) /
                       hist[hist["status"].isin(["completed", "cancelled"])].groupby("customer_id").size().reindex(feat.index))
feat = feat.join(c.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns=["signup_date"])
y = pd.Series([0 if cid in active_sept else 1 for cid in feat.index], index=feat.index, name="lapsed")

print("cohort:", len(feat), "| lapsed:", int(y.sum()), "(%.2f%%)" % (100 * y.mean()))
assert o.loc[o["order_ts"] >= CUTOFF, "order_id"].isin(hist["order_id"]).sum() == 0  # no leakage of Sept orders into features

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
assert not feat.isna().any().any(), "missing values in features"
print("no feature column has a missing value: True")

X_tr, X_te, y_tr, y_te = train_test_split(feat, y, test_size=0.2, stratify=y, random_state=42)
print("train/test:", len(X_tr), len(X_te), "| train lapsed", int(y_tr.sum()), "| test lapsed", int(y_te.sum()))

base_pred = np.zeros(len(y_te), dtype=int)
print("baseline always 'not lapsed': accuracy %.3f F1 %.3f" %
      (accuracy_score(y_te, base_pred), f1_score(y_te, base_pred, zero_division=0)))


def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])


def metrics(model, X, y_true):
    pr = model.predict(X)
    return (precision_score(y_true, pr, zero_division=0), recall_score(y_true, pr, zero_division=0),
            f1_score(y_true, pr, zero_division=0), confusion_matrix(y_true, pr))


# ---------------- task 5/6: tuning (the whole Pipeline is refit inside every CV fold => no leakage)
lr_gs = GridSearchCV(Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))]),
                     {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr_gs.fit(X_tr, y_tr)
rf_gs = GridSearchCV(Pipeline([("prep", prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
rf_gs.fit(X_tr, y_tr)
print("\nLogReg best:", lr_gs.best_params_, "CV F1 %.4f" % lr_gs.best_score_)
print("RF best:", rf_gs.best_params_, "CV F1 %.4f" % rf_gs.best_score_)

lr_best, rf_best = lr_gs.best_estimator_, rf_gs.best_estimator_
sc = lr_best.named_steps["prep"].named_transformers_["num"]
print("\nlearned (train-only) mean of frequency: %.4f | full-cohort mean of frequency: %.4f" %
      (sc.mean_[NUM.index("frequency")], feat["frequency"].mean()))

res = {}
for name, m in [("LogReg", lr_best), ("RandomForest", rf_best)]:
    pr, rc, f1, cm = metrics(m, X_te, y_te)
    res[name] = f1
    print(f"\n{name} test: precision {pr:.3f} recall {rc:.3f} F1 {f1:.3f}\nconfusion [[TN FP],[FN TP]]:\n{cm}")

# Model selection uses cross-validated F1 only (never the test set).
selected_name = "RandomForest" if rf_gs.best_score_ >= lr_gs.best_score_ else "LogReg"
selected = rf_best if selected_name == "RandomForest" else lr_best
print("\nselected by CV F1:", selected_name)

# ---------------- task 7: class-balance experiment (oversample TRAIN rows only)
lapsed_tr = X_tr[y_tr == 1]
n_add = int((y_tr == 0).sum() - (y_tr == 1).sum())
extra = lapsed_tr.sample(n=n_add, replace=True, random_state=42)
X_bal = pd.concat([X_tr, extra]); y_bal = pd.concat([y_tr, pd.Series(1, index=extra.index)])
print("\nbalanced train counts:", y_bal.value_counts().to_dict())
lr_bal = clone(lr_best).fit(X_bal, y_bal)
p0, r0, f0, _ = metrics(lr_best, X_te, y_te)
p1, r1, f1b, cmb = metrics(lr_bal, X_te, y_te)
print("LogReg unbalanced: precision %.3f recall %.3f F1 %.3f" % (p0, r0, f0))
print("LogReg balanced  : precision %.3f recall %.3f F1 %.3f\n%s" % (p1, r1, f1b, cmb))

# ---------------- task 8: interpretation, persistence, retention list
names = rf_best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\ntop-5 RF feature importances:")
print(imp.head(5).round(4).to_string())
coef = pd.Series(lr_best.named_steps["clf"].coef_[0], index=lr_best.named_steps["prep"].get_feature_names_out())
print("\nlogistic coefficients (standardised):")
for f in ["num__recency", "num__frequency", "num__monetary", "num__cancel_rate", "num__avg_delivery", "num__avg_discount"]:
    print(f, round(coef[f], 4))

joblib.dump(selected, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(selected.predict(X_te), re.predict(X_te)) and np.allclose(selected.predict_proba(X_te), re.predict_proba(X_te))
print("\nreloaded model reproduces test predictions exactly:", bool(same))
assert same

seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
prob = pd.Series(selected.predict_proba(X_te)[:, 1], index=X_te.index)
prem = prob[[cid for cid in prob.index if cid in seg.index and bool(seg.loc[cid, "is_premium"])]]
top = prem.sort_values(ascending=False).head(10)
ret = pd.DataFrame({"customer_id": top.index, "lapse_probability": top.values.round(4),
                    "actually_lapsed": y_te.loc[top.index].values})
ret.to_csv("outputs/retention_list.csv", index=False)
print("\npremium customers in test set:", len(prem), "| retention list rows:", len(ret),
      "| truly lapsed:", int(ret["actually_lapsed"].sum()))
print(ret.to_string(index=False))
