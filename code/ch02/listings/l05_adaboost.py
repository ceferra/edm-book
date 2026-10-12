import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y01 = make_classification(n_samples=2000, n_features=10, n_informative=5,
                             random_state=4)
y = 2 * y01 - 1                                   # AdaBoost uses labels in {-1, +1}
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)

w = np.full(len(y_tr), 1 / len(y_tr))             # uniform initial weights
stumps, alphas = [], []
for t in range(100):
    h = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr, sample_weight=w)
    miss = h.predict(X_tr) != y_tr
    eps = w[miss].sum()                           # weighted training error
    if eps == 0 or eps >= 0.5:                    # perfect, or no weak learner
        break
    alpha = 0.5 * np.log((1 - eps) / eps)         # weight of the new member
    w *= np.exp(np.where(miss, alpha, -alpha))    # mistakes gain weight
    w /= w.sum()
    stumps.append(h)
    alphas.append(alpha)
    if t < 3:
        print(f"round {t + 1}: weighted error={eps:.3f}  alpha={alpha:.3f}")

F = sum(a * h.predict(X_te) for a, h in zip(alphas, stumps))
print(f"our AdaBoost (100 stumps):  test accuracy={np.mean(np.sign(F) == y_te):.3f}")
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100,
                         random_state=0).fit(X_tr, y_tr)
print(f"scikit-learn's AdaBoost:    test accuracy={ada.score(X_te, y_te):.3f}")
print(f"the first stump:            test accuracy={stumps[0].score(X_te, y_te):.3f}")
mean_acc = np.mean([h.score(X_te, y_te) for h in stumps])
print(f"mean of the 100 stumps:      test accuracy={mean_acc:.3f}")
