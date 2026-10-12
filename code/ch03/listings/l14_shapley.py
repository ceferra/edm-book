from itertools import combinations
from math import factorial
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from apartments import FEATURES, apartments

def shapley(f, x, background):
    """Exact Shapley values; absent features take their values from background data."""
    p = len(x)
    def value(S):                                   # v(S) = E[f(x_S, X_rest)]
        Z = background.copy()
        Z[:, list(S)] = x[list(S)]
        return f(Z).mean()
    phi = np.zeros(p)
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for size in range(p):
            for S in combinations(others, size):
                weight = factorial(size) * factorial(p - size - 1) / factorial(p)
                phi[j] += weight * (value(S + (j,)) - value(S))
    return phi

X, y = apartments()
model = GradientBoostingRegressor(random_state=0).fit(X, y)
x = np.array([1.0, 0.0, 50.0, 2.0])
bg = X[np.random.default_rng(1).choice(len(X), 300, replace=False)]
phi = shapley(model.predict, x, bg)
base, pred = model.predict(bg).mean(), model.predict(x[None])[0]
for name, v in zip(FEATURES, phi):
    print(f"{name:12s} {v:+7.1f}")
print(f"average {base:.1f} + contributions {phi.sum():+.1f} = {base + phi.sum():.1f}"
      f"   (prediction {pred:.1f})")
