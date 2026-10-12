from sklearn.datasets import make_classification
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=3000, n_features=20, n_informative=10, n_redundant=4,
                           flip_y=0.05, random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
models = {
    "single tree": DecisionTreeClassifier(random_state=0),
    "bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=200,
                                 oob_score=True, random_state=0),
    "random forest": RandomForestClassifier(n_estimators=200, oob_score=True,
                                            random_state=0),
    "extra trees": ExtraTreesClassifier(n_estimators=200, bootstrap=True, oob_score=True,
                                        random_state=0)}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    oob = f"   OOB={m.oob_score_:.3f}" if hasattr(m, "oob_score_") else ""
    print(f"{name:14s} test accuracy={m.score(X_te, y_te):.3f}{oob}")
