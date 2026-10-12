import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width bins."""
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return sum(np.mean(b == k) * abs(y[b == k].mean() - p[b == k].mean())
               for k in range(n_bins) if np.any(b == k))

X, y = make_classification(n_samples=40000, n_features=20, n_informative=2,
                           n_redundant=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
models = {"NB": GaussianNB(),
          "NB + Platt": CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5),
          "NB + isotonic": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)}
for name, m in models.items():
    p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    print(f"{name:14s} Brier={brier_score_loss(y_te, p):.4f}  "
          f"LogLoss={log_loss(y_te, p):.4f}  ECE={ece(y_te, p):.4f}")
