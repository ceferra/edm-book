import hashlib
import io
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "d20fcb6402ae34e653d4513b00f39257bb37ed7f/german.csv")
raw = urlopen(URL).read()
data_hash = hashlib.sha256(raw).hexdigest()
data = pd.read_csv(io.BytesIO(raw), header=None)
y = (data[20] == 2).astype(int)                                  # 1 = bad applicant
X = data.drop(columns=20).rename(columns={1: "duration", 7: "rate", 12: "age"})
X.columns = X.columns.astype(str)
X = X.astype({c: "category" for c in X.select_dtypes(exclude="number").columns})
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=1)

model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=8,
                                       min_samples_leaf=20, random_state=0,
                                       monotonic_cst={"duration": 1, "rate": 1})
model.fit(X_tr, y_tr)
p = model.predict_proba(X_te)[:, 1]
approve = p < 1 / 6                         # a bad loan costs 5 times a lost good one

rng = np.random.default_rng(0)              # bootstrap interval for the AUC
idx = [rng.integers(0, len(p), len(p)) for _ in range(1000)]
boot = [roc_auc_score(y_te.iloc[i], p[i]) for i in idx]
lo, hi = np.percentile(boot, [2.5, 97.5])

rows = []
for name, mask in [("under 25", X_te.age < 25), ("25 or over", X_te.age >= 25)]:
    auc = roc_auc_score(y_te[mask], p[mask])
    rows.append(f"| {name:10s} | {mask.sum():3d} | {auc:.3f} "
                f"| {approve[mask].mean():.0%} | {y_te[mask & approve].mean():.0%} |")

card = f"""# Model card: credit risk score, version 1.0, trained 2026-10-29
Model: gradient boosting, risk increasing with duration and instalment rate.
Data: German credit, 1000 applicants, SHA-256 {data_hash[:12]}...; 600 train / 400 test.
Intended use: support for credit analysts; the decision stays with a person.
Not for: automatic rejection, other countries, other credit products.
Decision rule: approve if P(bad) < 1/6 (a bad loan costs five times a lost good one).
Test AUC: {roc_auc_score(y_te, p):.3f} (95% bootstrap interval {lo:.3f} to {hi:.3f}).
| group      |   n |   AUC | approved | bad among approved |
{chr(10).join(rows)}
Limitations: one German bank, data published in 1994; small test set (400);
outcomes known only for granted loans; the data include age, sex and marital status.
Monitoring: PSI of inputs and scores monthly; default rate of approved loans at 6 and
12 months; a random 5% of applications decided without the score (holdout)."""
print(card)
