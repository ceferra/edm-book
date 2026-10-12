import numpy as np
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

rng = np.random.default_rng(0)
LR = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
RF = RandomForestClassifier(100, random_state=0, n_jobs=-1)

def random_task(rng):
    """A random binary classification data set with varied size, noise and shape."""
    n, d = int(rng.integers(100, 600)), int(rng.integers(2, 40))
    inf = int(min(d, rng.integers(2, 6)))
    clusters = int(min(rng.integers(1, 4), 2 ** inf // 2))
    return make_classification(n_samples=n, n_features=d, n_informative=inf,
                               n_redundant=0, n_clusters_per_class=clusters,
                               class_sep=rng.uniform(0.5, 2), flip_y=rng.uniform(0, 0.1),
                               random_state=int(rng.integers(1e6)))

def meta_features(X, y):
    """Simple, statistical and landmarking meta-features (Pfahringer et al., 2000)."""
    n, d = X.shape
    land = lambda m: cross_val_score(m, X, y, cv=3).mean()
    return [np.log10(n), np.log10(d), np.log10(n / d), min(y.mean(), 1 - y.mean()),
            land(LinearDiscriminantAnalysis()), land(DecisionTreeClassifier(max_depth=1)),
            land(make_pipeline(StandardScaler(), KNeighborsClassifier(1)))]

NAMES = ["log_n", "log_d", "log_n/d", "minority", "lm_linear", "lm_stump", "lm_1nn"]
tasks = [random_task(rng) for _ in range(120)]
M = np.array([meta_features(X, y) for X, y in tasks])
acc = np.array([[cross_val_score(m, X, y, cv=3).mean() for m in (LR, RF)]
                for X, y in tasks])
winner = (acc[:, 1] > acc[:, 0]).astype(int)                 # 1 if the forest is better
meta = DecisionTreeClassifier(max_depth=2, random_state=0)
# leave one task out: the meta-model never sees the task it chooses an algorithm for
choice = cross_val_predict(meta, M, winner, cv=LeaveOneOut())
chosen = acc[np.arange(len(acc)), choice]
print(f"the forest wins on {winner.mean():.0%} of the 120 tasks")
print(f"meta-model picks the better algorithm on {np.mean(choice == winner):.0%} of them")
print(f"it chooses the logistic regression for {np.sum(choice == 0)} tasks")
for name, a in [("always logistic", acc[:, 0]), ("always forest", acc[:, 1]),
                ("meta-learned choice", chosen), ("oracle", acc.max(axis=1))]:
    regret = acc.max(axis=1) - a                  # accuracy lost against the best choice
    print(f"{name:20s} mean accuracy {a.mean():.4f}   mean regret {regret.mean():.4f}"
          f"   worst regret {regret.max():.3f}")
print(export_text(meta.fit(M, winner), feature_names=NAMES), end="")
