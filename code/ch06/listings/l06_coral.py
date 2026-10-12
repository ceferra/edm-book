import numpy as np
from scipy.linalg import sqrtm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

X, y = make_classification(n_samples=6000, n_features=10, n_informative=6, random_state=5)
X_s, X_t, y_s, y_t = train_test_split(X, y, test_size=0.5, random_state=0)
rng = np.random.default_rng(5)
gain = np.diag(rng.uniform(0.5, 2.0, 10))                  # recalibrated sensors...
mix = np.eye(10) + 0.3 * rng.normal(size=(10, 10))              # ...with cross-talk
X_t = X_t @ gain @ mix + rng.normal(0, 1, 10)                    # ...and offsets

def coral(X_s, X_t, eps=1.0):
    """CORAL (Sun et al., 2016): give the source the mean and (regularised) covariance of
    the target, approximately."""
    I = np.eye(X_s.shape[1])
    C_s, C_t = np.cov(X_s, rowvar=False) + eps * I, np.cov(X_t, rowvar=False) + eps * I
    recolour = np.real(np.linalg.inv(sqrtm(C_s)) @ sqrtm(C_t))   # whiten, then colour
    return (X_s - X_s.mean(0)) @ recolour + X_t.mean(0)

standardise = lambda Z: (Z - Z.mean(0)) / Z.std(0)
model = lambda: LogisticRegression(max_iter=2000)
print(f"no adaptation             {model().fit(X_s, y_s).score(X_t, y_t):.3f}")
per_domain = model().fit(standardise(X_s), y_s).score(standardise(X_t), y_t)
print(f"standardise each domain   {per_domain:.3f}")
print(f"CORAL                     {model().fit(coral(X_s, X_t), y_s).score(X_t, y_t):.3f}")
print(f"with target labels (CV)   {cross_val_score(model(), X_t, y_t, cv=5).mean():.3f}")
