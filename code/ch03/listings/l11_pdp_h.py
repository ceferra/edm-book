import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
model = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                  random_state=0).fit(X_tr, y_tr)

def pd_at(X, cols, values):
    """Partial dependence: average prediction with the columns `cols` set to `values`."""
    Xv = X.copy()
    Xv[:, cols] = values
    return model.predict(Xv).mean()

def h2(X, j, k):
    """Friedman's H^2 for features j and k, from centred partial dependence functions."""
    pd = lambda cols: np.array([pd_at(X, cols, x[cols]) for x in X])
    pj, pk, pjk = pd([j]), pd([k]), pd([j, k])
    pj, pk, pjk = pj - pj.mean(), pk - pk.mean(), pjk - pjk.mean()
    return np.sum((pjk - pj - pk) ** 2) / np.sum(pjk ** 2)

S = X_te[:120]
print("PD of x4 at 0, 0.5, 1:", " ".join(f"{pd_at(S, [3], v):.2f}" for v in (0, 0.5, 1)))
for j, k in [(0, 1), (0, 3), (3, 4)]:
    print(f"H^2(x{j + 1}, x{k + 1}) = {h2(S, j, k):.3f}")
