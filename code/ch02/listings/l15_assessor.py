import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)

def sample(n, spread=1.0):
    """Five features; the target depends on x1 squared, which a linear model misses."""
    X = rng.normal(size=(n, 5))
    X[:, 1] *= spread                            # spread > 1: more extreme values of x1
    logit = 2 * X[:, 0] + 1.5 * (X[:, 1] ** 2 - 1) + 0.5 * X[:, 2]
    return X, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

X_tr, y_tr = sample(3000)                        # trains the model
X_as, y_as = sample(3000)                        # trains the assessor (unseen by model)
X_te, y_te = sample(3000)                        # tests both
X_new, y_new = sample(3000, spread=1.6)          # deployment data after a covariate shift

model = LogisticRegression().fit(X_tr, y_tr)
right = lambda X, y: (model.predict(X) == y).astype(int)
confidence = lambda X: model.predict_proba(X).max(axis=1)
assessor = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0)
assessor.fit(X_as, right(X_as, y_as))            # learns where the model is right
p_right = lambda X: assessor.predict_proba(X)[:, 1]

r_te = right(X_te, y_te)
print(f"accuracy of the model on the test data: {r_te.mean():.3f}")
for name, score in [("the model's confidence", confidence(X_te)),
                    ("the assessor", p_right(X_te))]:
    keep = score >= np.quantile(score, 0.2)      # reject the 20% least likely to be right
    print(f"ranking by {name:22s} AUC {roc_auc_score(r_te, score):.3f}, "
          f"accuracy on the 80% kept {r_te[keep].mean():.3f}")
print(f"after the shift, true accuracy {right(X_new, y_new).mean():.3f}; "
      "estimated without labels:")
print(f"  test accuracy {r_te.mean():.3f}, "
      f"mean confidence {confidence(X_new).mean():.3f}, "
      f"assessor {p_right(X_new).mean():.3f}")
