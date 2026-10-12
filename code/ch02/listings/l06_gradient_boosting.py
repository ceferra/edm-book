from sklearn.datasets import make_classification
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=20000, n_features=30, n_informative=12, n_redundant=6,
                           flip_y=0.05, class_sep=0.8, random_state=2)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
models = {
    "random forest (300 trees)": RandomForestClassifier(300, n_jobs=-1, random_state=0),
    "gradient boosting (300 trees)": GradientBoostingClassifier(n_estimators=300,
                                                                random_state=0),
    "histogram GB + early stopping": HistGradientBoostingClassifier(
        max_iter=2000, early_stopping=True, n_iter_no_change=30, random_state=0)}
for name, m in models.items():
    auc = roc_auc_score(y_te, m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
    used = f"   (trees used: {m.n_iter_})" if hasattr(m, "n_iter_") else ""
    print(f"{name:30s} AUC={auc:.4f}{used}")
