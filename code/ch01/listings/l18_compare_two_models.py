import numpy as np
from scipy.stats import binomtest, t
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                           random_state=1)
A, B = LogisticRegression(max_iter=1000), GaussianNB()

# McNemar's exact test on a single test set
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
ok_a = A.fit(X_tr, y_tr).predict(X_te) == y_te
ok_b = B.fit(X_tr, y_tr).predict(X_te) == y_te
b, c = np.sum(ok_a & ~ok_b), np.sum(~ok_a & ok_b)
print(f"McNemar: b={b}, c={c}, exact p-value = {binomtest(b, b + c).pvalue:.4f}")

# Corrected resampled t-test (Nadeau & Bengio, 2003) on 10 x 10-fold CV
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
d = cross_val_score(A, X, y, cv=cv) - cross_val_score(B, X, y, cv=cv)
J, ratio = len(d), 1 / 9                  # n_test / n_train in 10-fold CV
t_c = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
t_naive = d.mean() / np.sqrt(d.var(ddof=1) / J)
print(f"mean difference = {d.mean():.4f}")
print(f"naive t = {t_naive:.2f} (p = {2 * t.sf(abs(t_naive), J - 1):.1e})")
print(f"corrected t = {t_c:.2f} (p = {2 * t.sf(abs(t_c), J - 1):.4f})")
