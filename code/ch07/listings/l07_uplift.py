import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

rng = np.random.default_rng(7)
n = 40_000
age, freq, web = rng.integers(18, 85, n), rng.poisson(3, n), rng.random(n) < 0.6
p0 = 1 / (1 + np.exp(-(-3.5 + 0.45 * freq + 0.3 * web)))      # buys without contact
positive_seg = (age < 35) & (freq <= 2)          # contact raises the purchase rate
negative_seg = (age >= 60) & (freq >= 5)          # contact lowers it
# the effect of the contact on the probability of buying
tau = np.where(positive_seg, 0.08, np.where(negative_seg, -0.06, 0.005))
X = np.c_[age, freq, web]
T = rng.random(n) < 0.5                                         # a randomised campaign
y = (rng.random(n) < np.clip(p0 + T * tau, 0, 1)).astype(int)
train, test = np.arange(n) < n // 2, np.arange(n) >= n // 2

gb = lambda: GradientBoostingClassifier(random_state=0)
# classic: who buys if mailed
response = gb().fit(X[train & T], y[train & T])
m1, m0 = gb().fit(X[train & T], y[train & T]), gb().fit(X[train & ~T], y[train & ~T])
scores = {"random": rng.random(n), "response model": response.predict_proba(X)[:, 1],
          "uplift (two models)": m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1]}
idx = np.flatnonzero(test)
print("top 20% by             true effect   estimated [95% CI]      + segment  - segment")
for name, s in scores.items():
    top = idx[np.argsort(-s[idx])[: len(idx) // 5]]            # the targeted customers
    treated, control = top[T[top]], top[~T[top]]               # randomised, so comparable
    effect = y[treated].mean() - y[control].mean()
    se = np.sqrt(y[treated].var(ddof=1) / len(treated)
                 + y[control].var(ddof=1) / len(control))
    lo, hi = 1000 * (effect - 1.96 * se), 1000 * (effect + 1.96 * se)
    print(f"{name:22s} {1000 * tau[top].mean():+7.1f}   {1000 * effect:+7.1f} "
          f"[{lo:+6.1f}, {hi:+6.1f}]      {positive_seg[top].mean():4.0%}      "
          f"{negative_seg[top].mean():4.0%}")
