import numpy as np
from sklearn.linear_model import LogisticRegression

def emails2d(n, seed):                        # two features that summarise each e-mail
    rng = np.random.default_rng(seed)
    X = np.r_[rng.normal([-1.2, -1.2], 1.0, (n, 2)), rng.normal([1.2, 1.2], 1.0, (n, 2))]
    return X, np.r_[np.zeros(n, int), np.ones(n, int)]                # 1 = spam

X, y = emails2d(1000, 0)
X_test, y_test = emails2d(5000, 1)
print("chaff   spam caught   ham flagged   uncertain (0.4 < p < 0.6)")
for frac in (0.0, 0.05, 0.1, 0.2):
    k = int(frac * len(y))
    # chaff: spam-like points labelled as ham (a larger attack adds to a smaller one)
    chaff = np.random.default_rng(2).normal([1.0, 1.0], 0.35, (k, 2))
    model = LogisticRegression().fit(np.r_[X, chaff], np.r_[y, np.zeros(k, int)])
    p = model.predict_proba(X_test)[:, 1]
    caught, flagged = np.mean(p[y_test == 1] >= 0.5), np.mean(p[y_test == 0] >= 0.5)
    band = np.mean((p > 0.4) & (p < 0.6))
    print(f"{frac:4.0%}      {caught:.3f}         {flagged:.3f}         {band:.3f}")
