import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class DelegatingClassifier:
    """Two-step delegation (after Ferri, Flach and Hernandez-Orallo, 2004).
    The first model delegates its least confident fraction of the examples. If
    specialise=True the second model is trained only on the delegated examples; if
    rebound > 0, it returns its own least confident fraction to the first model."""
    def __init__(self, first, second, delegate=0.3, specialise=True, rebound=0.0):
        self.first, self.second, self.delegate = first, second, delegate
        self.specialise, self.rebound = specialise, rebound

    def fit(self, X, y):
        conf = self.first.fit(X, y).predict_proba(X).max(axis=1)
        self.t1_ = np.quantile(conf, self.delegate)
        hard = conf <= self.t1_                               # delegated examples
        Xs, ys = (X[hard], y[hard]) if self.specialise else (X, y)
        conf2 = self.second.fit(Xs, ys).predict_proba(X[hard]).max(axis=1)
        self.t2_ = np.quantile(conf2, self.rebound) if self.rebound else -np.inf
        return self

    def predict(self, X):
        """The first model sees every example, the second only the delegated ones."""
        P1 = self.first.predict_proba(X)
        pred = self.first.classes_[P1.argmax(axis=1)]
        idx = np.flatnonzero(P1.max(axis=1) <= self.t1_)          # delegated examples
        self.n_second_ = len(idx)
        if len(idx):
            P2 = self.second.predict_proba(X[idx])
            take = P2.max(axis=1) > self.t2_                       # else: round rebound
            pred[idx[take]] = self.second.classes_[P2[take].argmax(axis=1)]
        return pred

X, y = make_classification(n_samples=6000, n_features=10, n_informative=6, n_redundant=2,
                           flip_y=0.03, class_sep=1.0, random_state=8)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
tree = lambda: DecisionTreeClassifier(max_depth=4, random_state=0)
forest = lambda: RandomForestClassifier(200, random_state=0)
systems = {"small tree alone": tree(),
           "specialist (two small trees)": DelegatingClassifier(tree(), tree()),
           "specialist + round rebound": DelegatingClassifier(tree(), tree(), rebound=0.3),
           "cascade: small tree -> forest":
               DelegatingClassifier(tree(), forest(), specialise=False),
           "random forest alone": forest()}
for name, m in systems.items():
    acc = np.mean(m.fit(X_tr, y_tr).predict(X_te) == y_te)
    calls = ""
    if hasattr(m, "n_second_"):                       # delegating systems
        calls = f"  (second model: {m.n_second_} of {len(y_te)})"
    print(f"{name:30s} accuracy={acc:.3f}{calls}")
