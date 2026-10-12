import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(2)
X = rng.uniform(0, 10, (8000, 1))
y = X[:, 0] * np.sin(X[:, 0]) + rng.normal(0, 0.3 + 0.3 * X[:, 0])  # noisier for large x
idx = rng.permutation(len(y))
tr, cal, te = idx[:2000], idx[2000:4000], idx[4000:]   # train / calibration / test

alpha = 0.1                                           # target coverage 90%
model = GradientBoostingRegressor(random_state=0).fit(X[tr], y[tr])
scores = np.abs(y[cal] - model.predict(X[cal]))       # nonconformity scores
assert 0 < alpha < 1 and len(scores) > 0
k = int(np.ceil((len(cal) + 1) * (1 - alpha)))
q = np.inf if k > len(scores) else np.sort(scores)[k - 1]   # conformal quantile
pred = model.predict(X[te])
covered = (y[te] >= pred - q) & (y[te] <= pred + q)
print(f"q = {q:.3f};  interval width = {2 * q:.2f}")
print(f"empirical coverage = {covered.mean():.3f}  (target {1 - alpha:.2f})")
