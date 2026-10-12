import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(4)

def week_data(t, n=2000):
    """Data of week t: the true decision boundary rotates slowly (concept drift)."""
    X = rng.normal(size=(n, 2))
    w = np.array([np.cos(0.12 * t), np.sin(0.12 * t)])
    return X, (X @ w + 0.3 * rng.normal(size=n) > 0).astype(int)

def data_ok(X, y):
    """Data quality tests: finite values, no constant column, plausible label rate."""
    return bool(np.isfinite(X).all() and (X.std(axis=0) > 1e-6).all()
                and 0.2 < y.mean() < 0.8)

def fit(batches):
    return LogisticRegression().fit(np.vstack([b[0] for b in batches]),
                                    np.concatenate([b[1] for b in batches]))

initial = week_data(0)                               # one initial batch, shared
clean, everything = [initial], [initial]
static = blind = ct = fit(clean)                     # all start from the same model
history = []
print("week  static  no checks  continuous  decision of the continuous pipeline")
for t in range(1, 13):
    X, y = week_data(t)                              # this week's data, labelled later
    accs = [m.score(X, y) for m in (static, blind, ct)]
    if t == 6:
        y = np.zeros_like(y)                         # a faulty join loses the labels
    everything.append((X, y))
    blind = fit(everything[-3:])                     # retrain and deploy without checks
    if not data_ok(X, y):
        decision = "data test failed: keep the model, alert"
    else:
        clean.append((X, y))
        candidate = fit(clean[-3:])                  # train on the last three weeks
        X_val, y_val = week_data(t, n=1000)          # fresh labelled data of this week
        gain = candidate.score(X_val, y_val) - ct.score(X_val, y_val)
        promote = gain >= -0.005                     # tolerance on a point estimate
        decision = f"{'promote' if promote else 'keep'} ({gain:+.3f})"
        ct = candidate if promote else ct
    history.append((t, *accs, decision))
    print(f"{t:4d}   {accs[0]:.3f}    {accs[1]:.3f}      {accs[2]:.3f}    {decision}")
mean = np.mean([h[1:4] for h in history], axis=0)
print(f"mean   {mean[0]:.3f}    {mean[1]:.3f}      {mean[2]:.3f}")
