import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=4600, n_features=8, n_informative=5, n_redundant=2,
                           flip_y=0.03, class_sep=1.0, random_state=7)
X_tr, y_tr, X_te, y_te = X[:600], y[:600], X[600:], y[600:]
oracle = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr)

def mimic(oracle, X, n, noise=0.3, max_leaves=32, seed=0):
    """Learn a small tree that imitates the oracle on n synthetic examples."""
    rng = np.random.default_rng(seed)
    Xs = X[rng.integers(0, len(X), n)]                          # resample the inputs
    Xs = Xs + rng.normal(0, noise, Xs.shape) * X.std(axis=0)    # and perturb them
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=0)
    return tree.fit(Xs, oracle.predict(Xs))                     # labels from the oracle

direct = DecisionTreeClassifier(max_leaf_nodes=32, random_state=0).fit(X_tr, y_tr)
mimetic = mimic(oracle, X_tr, n=20000)
ref = oracle.predict(X_te)
for name, m in [("random forest (oracle)", oracle), ("32-leaf tree, original data", direct),
                ("32-leaf mimetic tree", mimetic)]:
    print(f"{name:28s} accuracy={m.score(X_te, y_te):.3f}  "
          f"fidelity={np.mean(m.predict(X_te) == ref):.3f}")
