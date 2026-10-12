import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from loans import FEATURES, loan_data

X, y = loan_data()
model = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
f = lambda Z: model.predict_proba(Z)[:, 1]                    # P(repay)
mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)      # median absolute deviation
lo, hi = X.min(axis=0), X.max(axis=0)                          # plausible ranges

def counterfactual(x, target=0.5, free=(0, 1), lambdas=(1, 10, 100, 1000, 10000), tol=0.01):
    """Wachter et al.: minimise lam*(f(x')-target)^2 + MAD-weighted L1 distance to x."""
    free = list(free)
    def loss(z, lam):
        xc = x.copy()
        xc[free] = z
        out = np.sum(np.maximum(lo - xc, 0) + np.maximum(xc - hi, 0))   # stay plausible
        dist = np.sum(np.abs(xc - x)[free] / mad[free])          # MAD-weighted L1 distance
        return lam * (f(xc[None])[0] - target) ** 2 + dist + 1e3 * out
    z = x[free].copy()
    for lam in lambdas:                                        # raise lambda until close
        z = minimize(loss, z, args=(lam,), method="Nelder-Mead").x
        xc = x.copy()
        xc[free] = z
        if abs(f(xc[None])[0] - target) <= tol:
            break
    return xc

peter = np.array([30.0, 0.45, 5.0, 0.0])     # income, debt ratio, years in job, no default
# only income and debt ratio may change; aim just above the bank's threshold of 0.5
cf = counterfactual(peter, target=0.51, tol=0.005)
print(f"Peter:          P(repay) = {f(peter[None])[0]:.3f}")
print(f"counterfactual: P(repay) = {f(cf[None])[0]:.3f}")
print("decision:", "approved" if f(cf[None])[0] >= 0.5 else "still rejected")
for name, a, b in zip(FEATURES, peter, cf):
    if abs(a - b) > 1e-3:
        print(f"  {name}: {a:.3f} -> {b:.3f}")
