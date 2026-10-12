import numpy as np

rng = np.random.default_rng(14)
K, SIMS = 5, 200_000                                  # five equally spaced looks

def z_paths(drift):
    """z statistics at the K looks; `drift` is the expected z at the final look."""
    steps = rng.normal(drift / np.sqrt(K), 1.0, (SIMS, K))
    return np.cumsum(steps, axis=1) / np.sqrt(np.arange(1, K + 1))

def calibrate(shape, alpha=0.05):
    """Scale a boundary shape so that the overall false positive rate is alpha."""
    null = np.abs(z_paths(0.0))
    lo, hi = 1.0, 6.0
    for _ in range(40):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if np.mean((null > c * shape).any(axis=1)) > alpha else (lo, c)
    return hi * shape

looks = np.arange(1, K + 1)
designs = {"fixed (one test)": np.r_[np.full(K - 1, np.inf), 1.96],
           "Pocock": calibrate(np.ones(K)),
           "O'Brien-Fleming": calibrate(np.sqrt(K / looks))}
null, alt = np.abs(z_paths(0.0)), np.abs(z_paths(1.96 + 0.8416))   # 80% power if fixed
print("design             boundaries at the 5 looks          FPR    power   mean n")
for name, bound in designs.items():
    cross = alt > bound
    stop = np.where(cross.any(axis=1), cross.argmax(axis=1) + 1, K)
    bounds = " ".join(f"{b:5.2f}" if np.isfinite(b) else "    -" for b in bound)
    print(f"{name:17s}  {bounds}   {np.mean((null > bound).any(axis=1)):.3f}  "
          f"{cross.any(axis=1).mean():.3f}   {stop.mean() / K:.0%}")
