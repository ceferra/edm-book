import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

def transactions(n):
    X = rng.normal(size=(n, 5))
    logit = -4.5 + 1.5 * X[:, 0] + X[:, 1] - 0.8 * X[:, 2]
    return X, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)   # 1 = fraud

model = LogisticRegression().fit(*transactions(50_000))
X, y = transactions(200_000)                       # one month of production traffic
blocked = model.predict_proba(X)[:, 1] >= 0.2      # the model stops these transactions
holdout = rng.random(len(y)) < 0.02                # 2% of the traffic: never acted upon
observed = ~blocked | holdout                      # transactions whose outcome we see

def precision_recall(pred, truth):
    tp = np.sum(pred & (truth == 1))
    return tp / pred.sum(), tp / truth.sum()

p, r = precision_recall(blocked, y)
print(f"truth (unknowable): fraud rate {y.mean():.4f}, precision {p:.3f}, recall {r:.3f}")
print(f"fraud rate among the transactions we can observe: {y[observed].mean():.4f}")
p_h, r_h = precision_recall(blocked[holdout], y[holdout])
idx = np.flatnonzero(holdout)
boot = [precision_recall(blocked[j], y[j]) for j in
        (rng.choice(idx, len(idx)) for _ in range(1000))]
(pl, rl), (pu, ru) = np.percentile(boot, [2.5, 97.5], axis=0)
print(f"from the holdout: fraud rate {y[holdout].mean():.4f}")
print(f"  precision {p_h:.3f} [{pl:.2f}, {pu:.2f}] recall {r_h:.3f} [{rl:.2f}, {ru:.2f}]")
print(f"price of the holdout: {np.sum(blocked & holdout & (y == 1))} frauds let through")
