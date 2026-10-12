import numpy as np

# the experiment of the z-test listing
rng = np.random.default_rng(7)
n = 31_234
clicks_a, clicks_b = rng.binomial(n, 0.050), rng.binomial(n, 0.055)
draws = np.random.default_rng(15)
# Beta(1 + clicks, 1 + non-clicks): posterior of each CTR with a uniform prior
pa = draws.beta(1 + clicks_a, 1 + n - clicks_a, 200_000)
pb = draws.beta(1 + clicks_b, 1 + n - clicks_b, 200_000)
print(f"clicks A {clicks_a}, B {clicks_b} of {n:,} users each")
print(f"P(CTR of B > CTR of A) = {np.mean(pb > pa):.3f}")
lift = pb / pa - 1
lo, hi = np.percentile(lift, [2.5, 97.5])
print(f"relative lift: {np.median(lift):+.1%}, "
      f"95% credible interval [{lo:+.1%}, {hi:+.1%}]")
for choice, loss in [("A", np.maximum(pb - pa, 0)), ("B", np.maximum(pa - pb, 0))]:
    print(f"expected loss if we choose {choice}: {loss.mean():.5f} clicks per user")
