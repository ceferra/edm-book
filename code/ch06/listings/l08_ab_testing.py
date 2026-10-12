import numpy as np
from scipy.stats import norm

def sample_size(p1, p2, alpha=0.05, power=0.8):
    """Users per group to detect a change from p1 to p2 with a two-sided z-test."""
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    p = (p1 + p2) / 2
    num = z_a * np.sqrt(2 * p * (1 - p)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(num ** 2 / (p2 - p1) ** 2))

def z_test(clicks_a, n_a, clicks_b, n_b):
    """Two-proportion z-test; works element-wise on arrays."""
    pa, pb = clicks_a / n_a, clicks_b / n_b
    p = (clicks_a + clicks_b) / (n_a + n_b)
    z = (pb - pa) / np.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    return z, 2 * norm.sf(np.abs(z))

n = sample_size(0.050, 0.055)
print(f"users per group to detect a CTR of 5.5% against 5.0%: {n:,}")
rng = np.random.default_rng(7)
a, b = rng.binomial(n, 0.050), rng.binomial(n, 0.055)
z, p = z_test(a, n, b, n)
print(f"one experiment: CTR A {a / n:.4f}, CTR B {b / n:.4f}, z = {z:.2f}, p = {p:.4f}")

looks, reps = 20, 5000                       # A/A tests: no real difference at all
seg = n // looks
users = seg * np.arange(1, looks + 1)
end = peek = 0
for _ in range(reps):
    ca = np.cumsum(rng.binomial(seg, 0.05, looks))
    cb = np.cumsum(rng.binomial(seg, 0.05, looks))
    p = z_test(ca, users, cb, users)[1]
    end += p[-1] < 0.05                       # one test at the planned end
    peek += np.any(p < 0.05)                  # stop as soon as anything looks significant
print(f"false positives in A/A tests: {end / reps:.3f} testing once, "
      f"{peek / reps:.3f} peeking {looks} times")
