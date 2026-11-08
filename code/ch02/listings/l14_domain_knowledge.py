import hashlib
import io
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# The German credit data of the introduction, from the same pinned mirror
URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
       "d20fcb6402ae34e653d4513b00f39257bb37ed7f/german.csv")
SHA256 = "ec12a88b9fc14d74ba646ea0410cf7ff4533bec2eb61652f8ad76796bbfec017"
raw = urlopen(URL).read()
assert hashlib.sha256(raw).hexdigest() == SHA256, "the data file has changed"
data = pd.read_csv(io.BytesIO(raw), header=None)
y = (data[20] == 2).astype(int)                       # 1 = the applicant turned out bad
X = data.drop(columns=20).rename(columns={0: "checking", 1: "duration", 5: "savings",
                                          6: "employment", 7: "rate"})
X.columns = X.columns.astype(str)
X = X.astype({c: "category" for c in X.select_dtypes(exclude="number").columns})

def scorecard(X):
    """A manual model: risk points for rules that a credit analyst might write."""
    return (2 * (X.checking == "A11") + (X.checking == "A12")   # negative or low balance
            + X.duration / 12                                   # one point per year of loan
            + (X.savings == "A61") + (X.rate == 4)              # low savings, high instalment
            + X.employment.isin(["A71", "A72"])).astype(float)  # unemployed or new in the job

def with_card(X):
    return X.assign(card=scorecard(X))

def falls(risk, X):
    """Directional test: share of applicants whose risk falls if the loan is a year longer."""
    return np.mean(risk(X.assign(duration=X.duration + 12)) < risk(X))

def boosting(constraints=None):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=8,
                                          min_samples_leaf=20, monotonic_cst=constraints,
                                          random_state=0)

rising = {"duration": 1, "rate": 1}         # the risk may only grow with these features
systems = {"gradient boosting": (None, False), "  + scorecard as a feature": (None, True),
           "  + monotonic constraints": (rising, False),
           "  + both": (rising | {"card": 1}, True)}
results = {name: [] for name in ["scorecard alone", *systems]}
for seed in range(10):                                          # ten random splits
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y,
                                              random_state=seed)
    results["scorecard alone"].append((roc_auc_score(y_te, scorecard(X_te)),
                                       falls(scorecard, X_te)))
    for name, (constraints, card) in systems.items():
        prep = with_card if card else (lambda Z: Z)
        model = boosting(constraints).fit(prep(X_tr), y_tr)
        risk = lambda Z: model.predict_proba(prep(Z))[:, 1]
        results[name].append((roc_auc_score(y_te, risk(X_te)), falls(risk, X_te)))

print(f"{'':28s} AUC    risk falls with a longer loan")
for name, r in results.items():
    auc, fall = np.mean(r, axis=0)
    print(f"{name:28s} {auc:.3f}  {fall:6.1%}")
