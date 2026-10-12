import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

def wilson_ci(k, n, conf=0.95):
    """Wilson score interval for a proportion (e.g. accuracy = k/n)."""
    z, p = norm.ppf(0.5 + conf / 2), k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half

def bootstrap_ci(metric, y, s, B=2000, conf=0.95, seed=0):
    """Percentile bootstrap interval for any metric(y, s)."""
    rng, stats = np.random.default_rng(seed), []
    while len(stats) < B:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():          # the resample needs both classes
            stats.append(metric(y[i], s[i]))
    return np.quantile(stats, [(1 - conf) / 2, (1 + conf) / 2])

print("accuracy 170/200 = 0.850, 95%% Wilson CI: [%.3f, %.3f]" % wilson_ci(170, 200))
rng = np.random.default_rng(1)
y = np.r_[np.ones(60), np.zeros(140)]
s = np.r_[rng.normal(1.2, 1, 60), rng.normal(0, 1, 140)]
lo, hi = bootstrap_ci(roc_auc_score, y, s)
print(f"AUC = {roc_auc_score(y, s):.3f}, 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")
