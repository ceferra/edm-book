import numpy as np

rng = np.random.default_rng(13)
n = 20_000                                            # users per group
def group(effect):
    # activity score in the month before the test
    before = rng.gamma(2.0, 2.5, n)
    during = 1.0 + 0.8 * before + rng.normal(0, 2.0, n) + effect
    return before, during

xa, ya = group(0.0)
xb, yb = group(0.1)                                   # B adds 0.1 to the activity score
x, y = np.r_[xa, xb], np.r_[ya, yb]
theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)        # CUPED: Y - theta (X - mean X)
adj = lambda xs, ys: ys - theta * (xs - x.mean())

def summary(a, b):
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    return diff, 1.96 * se

for name, (a, b) in [("plain difference", (ya, yb)),
                     ("CUPED", (adj(xa, ya), adj(xb, yb)))]:
    diff, half = summary(a, b)
    print(f"{name:17s} effect {diff:+.3f} +/- {half:.3f}   (95% CI)")
rho = np.corrcoef(x, y)[0, 1]
print(f"correlation before/during {rho:.2f}: variance and sample size reduced by "
      f"{rho ** 2:.0%}")
