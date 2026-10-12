import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
rf = RandomForestRegressor(300, random_state=0).fit(X_tr, y_tr)
pi = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0)   # drop in R^2
for j in np.argsort(-pi.importances_mean)[:6]:
    print(f"x{j + 1:<2d} permutation={pi.importances_mean[j]:.3f} "
          f"+/- {pi.importances_std[j]:.3f}   impurity={rf.feature_importances_[j]:.3f}")
for depth in (2, 3, 5):        # global surrogate: a small tree that imitates the forest
    tree = DecisionTreeRegressor(max_depth=depth, random_state=0)
    tree.fit(X_tr, rf.predict(X_tr))
    print(f"surrogate tree of depth {depth}: R^2 with respect to the forest = "
          f"{r2_score(rf.predict(X_te), tree.predict(X_te)):.3f}")
