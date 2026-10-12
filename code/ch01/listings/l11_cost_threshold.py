import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

C_FP, C_FN = 100, 2000           # cost of a false alarm and of a missed positive
t_star = C_FP / (C_FP + C_FN)    # Bayes-optimal threshold for calibrated scores

X, y = make_classification(n_samples=40000, n_features=10, n_informative=5,
                           weights=[0.97], flip_y=0.01, random_state=3)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y,
                                          random_state=0)
p = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

def total_cost(y, yhat):
    return C_FP * np.sum((yhat == 1) & (y == 0)) + C_FN * np.sum((yhat == 0) & (y == 1))

for t in (0.5, t_star):
    yhat = (p >= t).astype(int)
    print(f"t={t:.4f}  FP={np.sum((yhat == 1) & (y_te == 0)):4d}  "
          f"FN={np.sum((yhat == 0) & (y_te == 1)):3d}  "
          f"cost={total_cost(y_te, yhat):>9,} EUR")
