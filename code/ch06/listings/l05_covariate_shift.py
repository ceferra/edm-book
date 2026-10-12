import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(4)
f = lambda x: -x + x ** 3                              # the true regression function
x_s = rng.normal(0.5, 0.5, 200)                        # training inputs
y_s = f(x_s) + rng.normal(0, 0.3, 200)
x_t = rng.normal(0.0, 0.3, 1000)                       # deployment inputs
y_t = f(x_t) + rng.normal(0, 0.3, 1000)                # labels: evaluation and reference
col = lambda x: x.reshape(-1, 1)                       # a misspecified linear model

w_true = norm.pdf(x_s, 0, 0.3) / norm.pdf(x_s, 0.5, 0.5)    # p_target(x) / p_source(x)
both = np.r_[x_s, x_t]
is_target = np.r_[np.zeros(200), np.ones(1000)]
domain = LogisticRegression().fit(np.c_[both, both ** 2], is_target)
q = domain.predict_proba(np.c_[x_s, x_s ** 2])[:, 1]   # P(deployment | x)
w_est = 200 / 1000 * q / (1 - q)                       # the same ratio, from a classifier
fit = lambda x, y, w=None: LinearRegression().fit(col(x), y, sample_weight=w)
fits = {"unweighted": fit(x_s, y_s), "weighted (estimated w)": fit(x_s, y_s, w_est),
        "weighted (true w)": fit(x_s, y_s, w_true),
        "deployment fit (resubstitution)": fit(x_t, y_t)}
for name, m in fits.items():
    mse = np.mean((m.predict(col(x_t)) - y_t) ** 2)
    print(f"{name:32s} slope {m.coef_[0]:+.2f}   MSE on deployment data {mse:.3f}")
ess = lambda w: w.sum() ** 2 / (w ** 2).sum()          # effective sample size
print(f"effective sample size: {ess(w_est):.0f} (estimated w), {ess(w_true):.0f} (true w)"
      " of 200")
