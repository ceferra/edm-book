import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=2300, n_features=20, n_informative=5, flip_y=0.1,
                           random_state=5)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=300, random_state=0)
rng = np.random.default_rng(0)

def random_config(rng):
    """A random point of the CASH space: an algorithm and its hyperparameters."""
    algo = rng.choice(["logistic", "knn", "tree", "forest", "svm"])
    scaled = lambda m: make_pipeline(StandardScaler(), m)
    if algo == "logistic":
        return scaled(LogisticRegression(C=10 ** rng.uniform(-3, 3)))
    if algo == "knn":
        return scaled(KNeighborsClassifier(int(rng.integers(1, 50))))
    if algo == "tree":
        return DecisionTreeClassifier(max_depth=int(rng.integers(1, 20)),
                                      min_samples_leaf=int(rng.integers(1, 20)),
                                      random_state=0)
    if algo == "forest":
        return RandomForestClassifier(int(rng.integers(10, 200)),
                                      max_features=float(rng.uniform(0.1, 1)),
                                      random_state=0)
    return scaled(SVC(C=10 ** rng.uniform(-2, 3), gamma=10 ** rng.uniform(-4, 1)))

configs = [random_config(rng) for _ in range(200)]
cv = np.array([cross_val_score(c, X_tr, y_tr, cv=5).mean() for c in configs])
test = np.array([c.fit(X_tr, y_tr).score(X_te, y_te) for c in configs])
print("budget  best CV accuracy  test accuracy of that configuration")
for B in (1, 5, 20, 50, 100, 200):          # random search with budget B
    picks = [rng.choice(200, B, replace=False) for _ in range(500)]
    best = [idx[np.argmax(cv[idx])] for idx in picks]
    print(f"{B:5d}       {cv[best].mean():.3f}              {test[best].mean():.3f}")
print(f"best test accuracy of any configuration: {test.max():.3f}")
