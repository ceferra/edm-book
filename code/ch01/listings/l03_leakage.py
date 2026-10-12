import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5000))    # 5000 pure-noise features
y = rng.integers(0, 2, size=100)    # random labels: true accuracy is 0.5
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
clf = LogisticRegression(max_iter=1000)

# WRONG: the features are selected looking at ALL the labels, then we do CV
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
wrong = cross_val_score(clf, X_sel, y, cv=cv)

# RIGHT: selection is part of the model, so it is refitted inside each fold
right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), clf), X, y, cv=cv)
print(f"Wrong CV accuracy: {wrong.mean():.2f}")
print(f"Right CV accuracy: {right.mean():.2f}")
