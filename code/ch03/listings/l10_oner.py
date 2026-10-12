import numpy as np
from loans import FEATURES, loan_data

def one_r(Xcat, y):
    """OneR (Holte, 1993): the best set of rules that uses a single feature."""
    best = None
    for j in range(Xcat.shape[1]):
        rule = {v: np.bincount(y[Xcat[:, j] == v], minlength=2).argmax()   # majority class
                for v in np.unique(Xcat[:, j])}
        acc = np.mean([rule[v] == t for v, t in zip(Xcat[:, j], y)])
        if best is None or acc > best[1]:
            best = (j, acc, rule)
    return best

X, y = loan_data()
cuts = [np.quantile(X[:, j], [0.25, 0.5, 0.75]) for j in range(3)]    # quartiles
Xcat = np.c_[np.column_stack([np.digitize(X[:, j], cuts[j]) for j in range(3)]), X[:, 3]]
j, acc, rule = one_r(Xcat, y)
base = max(y.mean(), 1 - y.mean())
print(f"OneR uses '{FEATURES[j]}': accuracy {acc:.3f} "
      f"(predicting the majority: {base:.3f})")
for v, c in rule.items():
    outcome = "repays" if c else "defaults"
    print(f"  IF {FEATURES[j]} is in quartile {int(v) + 1} THEN {outcome}")
