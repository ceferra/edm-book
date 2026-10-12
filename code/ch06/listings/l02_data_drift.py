import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(1)

def window(n=5000, shift=0.0, sd3=1.0, rho=0.8):
    """Three features; x1 and x2 correlated with coefficient rho."""
    cov = np.array([[1, rho, 0], [rho, 1, 0], [0, 0, sd3 ** 2]])
    return rng.multivariate_normal([shift, 0, 0], cov, n)

def psi(ref, cur, bins=10):
    """Population stability index over the deciles of the reference window."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.maximum(np.histogram(ref, edges)[0] / len(ref), 1e-6)
    a = np.maximum(np.histogram(cur, edges)[0] / len(cur), 1e-6)
    return float(np.sum((a - e) * np.log(a / e)))

def domain_auc(ref, cur):
    """Can a classifier tell the two windows apart? An AUC of 0.5 means no."""
    X = np.vstack([ref, cur])
    d = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=0)
    return cross_val_score(clf, X, d, cv=5, scoring="roc_auc").mean()

reference = window()
scenarios = {"no drift": window(), "mean of x1 +0.1 sd": window(shift=0.1),
             "mean of x1 +0.5 sd": window(shift=0.5), "sd of x3 x1.5": window(sd3=1.5),
             "correlation 0.8 -> -0.8": window(rho=-0.8)}
print("scenario                  max KS (min p)    max PSI   domain AUC")
for name, current in scenarios.items():
    ks = [ks_2samp(reference[:, j], current[:, j]) for j in range(3)]
    stat, p = max(k.statistic for k in ks), min(k.pvalue for k in ks)
    worst_psi = max(psi(reference[:, j], current[:, j]) for j in range(3))
    print(f"{name:24s}  {stat:.3f} ({p:7.1e})   {worst_psi:.3f}     "
          f"{domain_auc(reference, current):.3f}")
