import numpy as np

rng = np.random.default_rng(0)
n = 1_000_000

def heart_model(do_W=None):
    """Structural causal model: exercise X, overweight W and heart disease H."""
    U1, U2, U3 = rng.random(n) < 1 / 2, rng.random(n) < 1 / 3, rng.random(n) < 1 / 3
    X = U1
    W = np.where(X, False, U2) if do_W is None else np.full(n, do_W)
    H = np.where(X, False, U3)
    return X, W, H

X, W, H = heart_model()
print(f"P(H = 1)              = {H.mean():.3f}   (exact 1/6 = 0.167)")
print(f"P(H = 1 | W = 1)      = {H[W].mean():.3f}   (exact 1/3 = 0.333)")
_, _, H_do = heart_model(do_W=True)
print(f"P(H = 1 | do(W := 1)) = {H_do.mean():.3f}   (W does not cause H)")
