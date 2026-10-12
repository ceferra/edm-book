import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
X = rng.uniform(0, 1, (400, 1));     y = f(X[:, 0]) + rng.normal(0, 0.3, 400)
X_te = rng.uniform(0, 1, (2000, 1)); y_te = f(X_te[:, 0]) + rng.normal(0, 0.3, 2000)

F = []
for t in range(10):                                    # 10 trees on bootstrap samples
    i = rng.integers(0, len(y), len(y))
    F.append(DecisionTreeRegressor(max_depth=6).fit(X[i], y[i]).predict(X_te))
F = np.array(F)
f_bar = F.mean(axis=0)                                 # the ensemble prediction

E = np.mean((f_bar - y_te) ** 2)                       # error of the ensemble
E_bar = np.mean((F - y_te) ** 2)                       # average error of the members
A_bar = np.mean((F - f_bar) ** 2)                      # average ambiguity (diversity)
print(f"ensemble error        E     = {E:.4f}")
print(f"average member error  E_bar = {E_bar:.4f}")
print(f"average ambiguity     A_bar = {A_bar:.4f}")
print(f"E_bar - A_bar               = {E_bar - A_bar:.4f}")
