from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline     # resamples ONLY the training folds
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

X, y = make_classification(n_samples=10000, n_features=10, n_informative=4,
                           weights=[0.97], flip_y=0.01, class_sep=1.5, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
lr = lambda **kw: LogisticRegression(max_iter=1000, **kw)
models = {"plain LR": lr(),
          "weighted LR": lr(class_weight="balanced"),
          "SMOTE + LR": make_pipeline(SMOTE(random_state=0), lr())}
scoring = ["balanced_accuracy", "f1", "average_precision", "neg_brier_score"]
for name, m in models.items():
    r = cross_validate(m, X, y, cv=cv, scoring=scoring)
    print(f"{name:11s} BA={r['test_balanced_accuracy'].mean():.3f} "
          f"F1={r['test_f1'].mean():.3f} AP={r['test_average_precision'].mean():.3f} "
          f"Brier={-r['test_neg_brier_score'].mean():.4f}")
