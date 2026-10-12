import hashlib
import numpy as np
from scipy.stats import chisquare

def variant(user, experiment, share_b=0.5):
    """Deterministic assignment: hash the user with an experiment-specific salt."""
    digest = hashlib.sha256(f"{experiment}:{user}".encode()).hexdigest()
    return "B" if int(digest[:8], 16) / 16 ** 8 < share_b else "A"

users = [f"user{i}" for i in range(100_000)]
first = np.array([variant(u, "new-ranker") for u in users])
second = np.array([variant(u, "new-colours") for u in users])
print("same user, same variant every time:", variant("user42", "new-ranker") == first[42])
both_b = np.mean((first == "B") & (second == "B"))
print(f"share in B in both experiments: {both_b:.3f} (independent: 0.250)")

def srm(n_a, n_b, share_b=0.5):
    """Sample ratio mismatch: do the observed counts match the intended split?"""
    total = n_a + n_b
    return chisquare([n_a, n_b], [total * (1 - share_b), total * share_b]).pvalue

alarm = lambda p: "SRM alarm" if p < 0.001 else "no alarm"   # strict: run on every test
n_a, n_b = np.sum(first == "A"), np.sum(first == "B")
p = srm(n_a, n_b)
print(f"logged users A {n_a:,}, B {n_b:,}: p-value {p:.4f}, {alarm(p)}")
lost = int(0.02 * n_b)                    # a bug in B loses 2% of its users' logs
p = srm(n_a, n_b - lost)
print(f"with 2% of B's logs lost: B {n_b - lost:,}, p-value {p:.1e}, {alarm(p)}")
