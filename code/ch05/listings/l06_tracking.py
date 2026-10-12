import csv
import datetime
import hashlib
import json
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

LOG = os.path.join(tempfile.mkdtemp(), "experiments.csv")   # versioned with the code
COLUMNS = ["date", "code_version", "model", "params", "train_size", "train_data",
           "test_size", "test_data", "cv_accuracy", "test_accuracy", "description"]

def fingerprint(*arrays):
    """Content hash of a data set (as DVC does): any change gives a new version."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:10]

def code_version():
    """Git commit of the code; outside a repository, a hash of this file."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:7]

def log_experiment(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as fh:
        writer = csv.DictWriter(fh, COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)

X, y = make_classification(n_samples=2000, n_features=20, n_informative=6, random_state=3)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
for Model, params, description in [
        (LogisticRegression, {"C": 0.1}, "baseline"),
        (LogisticRegression, {"C": 10.0}, "less regularisation"),
        (RandomForestClassifier, {"n_estimators": 200, "random_state": 0}, "non-linear")]:
    m = Model(**params)
    log_experiment(dict(
        date=datetime.date.today().isoformat(), code_version=code_version(),
        model=Model.__name__, params=json.dumps(params), description=description,
        train_size=len(y_tr), train_data=fingerprint(X_tr, y_tr),
        test_size=len(y_te), test_data=fingerprint(X_te, y_te),
        cv_accuracy=round(cross_val_score(m, X_tr, y_tr, cv=5).mean(), 4),
        test_accuracy=round(m.fit(X_tr, y_tr).score(X_te, y_te), 4)))

log = pd.read_csv(LOG)
table = log[["model", "params", "cv_accuracy", "test_accuracy"]]
table = table.rename(columns={"cv_accuracy": "cv", "test_accuracy": "test"})
print(table.to_string(index=False))
print(f"code version {log.code_version[0]}, training data {log.train_data[0]}")
X_tr[0, 0] += 1e-9                                   # change one value of the data...
print(f"after changing one value: training data {fingerprint(X_tr, y_tr)}")
