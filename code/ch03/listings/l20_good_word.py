import numpy as np
from sklearn.linear_model import LogisticRegression

def emails(n, seed):
    """Bag of words: words 0-14 are 'spammy', 15-29 'hammy' and 30-59 neutral."""
    rng = np.random.default_rng(seed)
    spam = np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)]
    p = np.where(spam[:, None] == 1, [0.30] * 15 + [0.05] * 15 + [0.1] * 30,
                 [0.03] * 15 + [0.25] * 15 + [0.1] * 30)
    return (rng.random((n, 60)) < p).astype(float), spam

def good_word_attack(model, x, budget=10):
    """Add, one at a time, the word that most lowers P(spam) (one query per candidate)."""
    x = x.copy()
    for k in range(budget + 1):
        if model.predict_proba(x[None])[0, 1] < 0.5:
            return k, x
        if k == budget:
            return None, x
        cand = np.flatnonzero(x == 0)
        Z = np.repeat(x[None], len(cand), axis=0)
        Z[np.arange(len(cand)), cand] = 1
        x[cand[np.argmin(model.predict_proba(Z)[:, 1])]] = 1

X, y = emails(4000, 0)
X_test, y_test = emails(2000, 1)
clf = LogisticRegression(max_iter=2000).fit(X, y)
adv = np.array([good_word_attack(clf, x)[1] for x in X[y == 1][:600]])
X_aug, y_aug = np.r_[X, adv], np.r_[y, np.ones(len(adv), int)]     # attacked spam is spam
robust = LogisticRegression(max_iter=2000).fit(X_aug, y_aug)
for name, m in [("standard filter", clf), ("adversarially trained", robust)]:
    caught = X_test[(y_test == 1) & (m.predict(X_test) == 1)][:300]
    need = [good_word_attack(m, x)[0] for x in caught]
    rates = [np.mean([k is not None and k <= b for k in need]) for b in (1, 3, 5, 10)]
    det = m.predict(X_test)[y_test == 1].mean()
    print(f"{name:22s} detection={det:.3f}  evasion with 1/3/5/10 words: "
          + " ".join(f"{r:.2f}" for r in rates))
