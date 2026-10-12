import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def member_predictions(ens, X):
    feats = getattr(ens, "estimators_features_", [slice(None)] * len(ens.estimators_))
    return np.array([e.predict(X[:, f]) for e, f in zip(ens.estimators_, feats)])

def pairwise_diversity(pred, y):
    """Average disagreement and Q statistic over all pairs of members."""
    ok = pred == y                                  # correct / incorrect outputs
    dis, Q = [], []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            n11, n00 = np.sum(ok[i] & ok[j]), np.sum(~ok[i] & ~ok[j])
            n10, n01 = np.sum(ok[i] & ~ok[j]), np.sum(~ok[i] & ok[j])
            dis.append((n01 + n10) / len(y))
            Q.append((n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10))
    return np.mean(dis), np.mean(Q)

X, y = make_classification(n_samples=3000, n_features=20, n_informative=10, n_redundant=4,
                           flip_y=0.05, random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
ensembles = {"bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=50,
                                          random_state=0),
             "random forest": RandomForestClassifier(50, random_state=0),
             "extra trees": ExtraTreesClassifier(50, random_state=0)}
for name, ens in ensembles.items():
    pred = member_predictions(ens.fit(X_tr, y_tr), X_te)
    dis, Q = pairwise_diversity(pred, y_te)
    print(f"{name:13s} member acc={np.mean(pred == y_te):.3f}  disagreement={dis:.3f}  "
          f"Q={Q:.3f}  ensemble acc={ens.score(X_te, y_te):.3f}")
