import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=20, n_informative=6, n_redundant=4,
                           flip_y=0.05, class_sep=0.8, random_state=5)
X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=2 / 3, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=1)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

alpha = 0.1                                                  # 90% prediction sets
s_cal = 1 - model.predict_proba(X_cal)[np.arange(len(y_cal)), y_cal]
k = int(np.ceil((len(y_cal) + 1) * (1 - alpha)))      # rank of the conformal quantile
q = np.inf if k > len(y_cal) else np.sort(s_cal)[k - 1]
sets = 1 - model.predict_proba(X_te) <= q                    # label j in the set?
size = sets.sum(axis=1)
one = size == 1                                              # predict; otherwise abstain
covered = np.mean(sets[np.arange(len(y_te)), y_te])
print(f"set coverage={covered:.3f} (target {1 - alpha:.2f})")
print(f"single-label sets={one.mean():.3f}  two labels (abstain)={np.mean(size == 2):.3f}")
acc_one = np.mean(sets[one].argmax(axis=1) == y_te[one])
print(f"accuracy on single-label sets={acc_one:.3f}")
