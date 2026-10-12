import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from apartments import FEATURES, apartments

def lime(f, X_train, x, n=5000, kernel_width=None, seed=0):
    """LIME for tabular data: a weighted linear model fitted to perturbed samples."""
    rng = np.random.default_rng(seed)
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    # samples from the data distribution
    Z = rng.normal(mu, sd, (n, len(x)))
    d = np.linalg.norm((Z - x) / sd, axis=1)              # standardised distance to x
    width = kernel_width or 0.75 * np.sqrt(len(x))
    w = np.exp(-d ** 2 / width ** 2)                      # proximity weights
    g = Ridge(alpha=1.0).fit((Z - mu) / sd, f(Z), sample_weight=w)
    return g.coef_, g.score((Z - mu) / sd, f(Z), sample_weight=w)

X, y = apartments()
model = GradientBoostingRegressor(random_state=0).fit(X, y)
x = np.array([1.0, 0.0, 50.0, 2.0])         # park nearby, cat forbidden, 50 m2, 2nd floor
for seed, width in [(0, None), (1, None), (2, None), (0, 0.5)]:
    coef, fid = lime(model.predict, X, x, kernel_width=width, seed=seed)
    print(f"seed {seed}, width {width or 1.5:.1f}: "
          + " ".join(f"{n}={c:.1f}" for n, c in zip(FEATURES, coef)) + f"  R2={fid:.2f}")
