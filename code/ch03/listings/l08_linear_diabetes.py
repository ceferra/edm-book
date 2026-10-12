import numpy as np
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
X, y = data.data, data.target
Xs = (X - X.mean()) / X.std()                            # standardised features
D = np.c_[np.ones(len(Xs)), Xs.values]                   # design matrix with intercept
beta = np.linalg.lstsq(D, y.values, rcond=None)[0]
res = y.values - D @ beta
n, p = D.shape
se = np.sqrt(np.diag(res @ res / (n - p) * np.linalg.inv(D.T @ D)))   # standard errors
r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
print(f"R2 = {r2:.3f}   adjusted R2 = {1 - (1 - r2) * (n - 1) / (n - p):.3f}")
for name, b, s in sorted(zip(X.columns, beta[1:], se[1:]), key=lambda t: -abs(t[1]))[:5]:
    print(f"{name:4s} weight={b:7.2f}   95% CI=[{b - 1.96 * s:7.2f}, {b + 1.96 * s:7.2f}]")
print(f"correlation between s1 and s2: {X.s1.corr(X.s2):.2f}")
