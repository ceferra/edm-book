import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

X, y = make_classification(n_samples=600, n_features=10, n_informative=4, n_redundant=2,
                           class_sep=0.8, flip_y=0.05, random_state=11)
lo, hi = np.array([-2.0, -4.0]), np.array([3.0, 1.0])      # log10 C and log10 gamma

def objective(z):
    """3-fold CV accuracy of an RBF SVM with C = 10^z[0] and gamma = 10^z[1]."""
    return cross_val_score(SVC(C=10 ** z[0], gamma=10 ** z[1]), X, y, cv=3).mean()

budget, rng = 25, np.random.default_rng(0)
grid = [np.array([a, b]) for a in np.linspace(lo[0], hi[0], 5)
        for b in np.linspace(lo[1], hi[1], 5)]                     # 5 x 5 grid
results = {"grid search": [(objective(z), z) for z in grid],
           "random search": [(objective(z), z) for z in rng.uniform(lo, hi, (budget, 2))]}
Z = list(rng.uniform(lo, hi, (5, 2)))                              # Bayesian optimisation
F = [objective(z) for z in Z]
while len(F) < budget:
    gp = GaussianProcessRegressor(Matern(nu=2.5), normalize_y=True, alpha=1e-4)
    gp.fit(np.array(Z), F)
    cand = rng.uniform(lo, hi, (2000, 2))
    mu, sd = gp.predict(cand, return_std=True)
    imp, sd = mu - max(F), np.maximum(sd, 1e-9)
    ei = imp * norm.cdf(imp / sd) + sd * norm.pdf(imp / sd)         # expected improvement
    Z.append(cand[np.argmax(ei)])
    F.append(objective(Z[-1]))
results["Bayesian optimisation"] = list(zip(F, Z))
for name, res in results.items():
    acc, z = max(res, key=lambda t: t[0])
    first = next(i for i, (a, _) in enumerate(res, 1) if a >= acc - 1e-12)
    print(f"{name:22s} best accuracy {acc:.3f} (evaluation {first:2d}) "
          f"at C=10^{z[0]:.2f}, gamma=10^{z[1]:.2f}")
