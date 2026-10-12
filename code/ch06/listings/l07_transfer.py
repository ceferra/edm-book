import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(6)
d = 20
w_source = rng.normal(size=d)                       # plenty of labelled data for this one
w_target = w_source + rng.normal(0, 0.5, d)         # a related concept, with few labels

def task(w, n):
    X = rng.normal(size=(n, d))
    return X, (rng.random(n) < 1 / (1 + np.exp(-(X @ w)))).astype(int)

X_s, y_s = task(w_source, 3000)
X_test, y_test = task(w_target, 5000)
LR = lambda: LogisticRegression(fit_intercept=False, max_iter=2000)
w0 = LR().fit(X_s, y_s).coef_[0]
zeros = lambda X: np.zeros_like(X)
augment = lambda X, target: np.c_[X, zeros(X), X] if target else np.c_[X, X, zeros(X)]

def towards_source(X, y, lam=5.0):
    """Logistic regression regularised towards the source weights (parameter transfer)."""
    def loss(w):
        z = X @ w
        return np.mean(np.logaddexp(0, z) - y * z) + lam * np.sum((w - w0) ** 2) / len(y)
    return minimize(loss, w0, method="L-BFGS-B").x

accuracy = lambda w: np.mean(((X_test @ w) > 0) == y_test)
methods = ["target", "source", "pooled", "augmented", "regularised"]
curves = {m: [] for m in methods}
sizes = [10, 20, 50, 100, 200, 500]
for n in sizes:
    scores = {m: [] for m in methods}
    for _ in range(10):
        X_t, y_t = task(w_target, n)
        while y_t.min() == y_t.max():
            X_t, y_t = task(w_target, n)
        scores["target"].append(accuracy(LR().fit(X_t, y_t).coef_[0]))
        scores["source"].append(accuracy(w0))
        pooled = LR().fit(np.r_[X_s, X_t], np.r_[y_s, y_t]).coef_[0]
        scores["pooled"].append(accuracy(pooled))
        aug = LR().fit(np.r_[augment(X_s, False), augment(X_t, True)], np.r_[y_s, y_t])
        scores["augmented"].append(np.mean(aug.predict(augment(X_test, True)) == y_test))
        scores["regularised"].append(accuracy(towards_source(X_t, y_t)))
    for m in methods:
        curves[m].append(np.mean(scores[m]))
print("target labels" + "".join(f"{m:>13s}" for m in methods))
for i, n in enumerate(sizes):
    print(f"{n:13d}" + "".join(f"{curves[m][i]:13.3f}" for m in methods))
