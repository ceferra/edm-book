import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict

rng = np.random.default_rng(3)
mu = np.array([1.0, 0.5, 0.0, -0.5, 0.8])          # how the positive class differs

def sample(n, prior):
    """Same class-conditional distributions p(x|y); only p(y) changes."""
    y = (rng.random(n) < prior).astype(int)
    return rng.normal(size=(n, 5)) + np.outer(y, mu), y

X_s, y_s = sample(5000, 0.5)                       # training data: balanced
model = LogisticRegression().fit(X_s, y_s)
X_t, y_t = sample(20000, 0.1)                      # deployment: 10% positives, no labels
p_t = model.predict_proba(X_t)[:, 1]

cc = np.mean(p_t >= 0.5)                           # classify and count
p_cv = cross_val_predict(LogisticRegression(), X_s, y_s, cv=5,
                         method="predict_proba")[:, 1]          # honest TPR and FPR
tpr, fpr = np.mean(p_cv[y_s == 1] >= 0.5), np.mean(p_cv[y_s == 0] >= 0.5)
acc = (cc - fpr) / (tpr - fpr)                     # adjusted classify and count
if not 0 <= acc <= 1:                              # not a valid prevalence: report it
    print(f"ACC = {acc:.3f} lies outside [0, 1]: the counts are not informative")
pcc = p_t.mean()                                   # probabilistic classify and count

def em_prior(p, prior_s, max_iter=1000, tol=1e-8):
    """Saerens et al. (2002): EM estimate of the new prior and of the posteriors."""
    prior = prior_s
    for _ in range(max_iter):                      # until the prior stops changing
        a = prior / prior_s * p
        post = a / (a + (1 - prior) / (1 - prior_s) * (1 - p))
        prior, old = post.mean(), prior
        if abs(prior - old) < tol:
            break
    return prior, post

em, post = em_prior(p_t, y_s.mean())
print(f"true prevalence {y_t.mean():.3f}; estimated: CC {cc:.3f}, ACC {acc:.3f}, "
      f"PCC {pcc:.3f}, EM {em:.3f}")
for name, q in [("unadjusted model", p_t), ("adjusted by EM", post)]:
    print(f"{name:17s} accuracy {np.mean((q >= 0.5) == y_t):.3f}   "
          f"Brier {brier_score_loss(y_t, q):.4f}   log loss {log_loss(y_t, q):.4f}")
