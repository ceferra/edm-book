from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# A small, noisy problem: 150 examples, 40 features (only 4 informative)
X, y = make_classification(n_samples=150, n_features=40, n_informative=4,
                           flip_y=0.15, class_sep=0.8, random_state=2)
inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)
grid = {"svc__C": [0.01, 0.1, 1, 10, 100, 1000],
        "svc__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1]}          # 30 candidate models
search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grid,
                      cv=inner, scoring="roc_auc")

search.fit(X, y)                     # model selection using all the data
print(f"Selected: {search.best_params_}")
print(f"Inner-CV AUC of the selected model (optimistic): {search.best_score_:.4f}")
# Nested CV: the WHOLE tuning procedure is repeated inside each outer fold
nested = cross_val_score(search, X, y, cv=outer, scoring="roc_auc")
print(f"Nested CV AUC (honest): {nested.mean():.4f} +/- {nested.std():.4f}")
