"""A synthetic customer database and a pilot mailing for the CRM examples (Unit 7)."""
import numpy as np

def customers(n=100_000, seed=6):
    rng = np.random.default_rng(seed)
    recency = rng.exponential(120, n)                  # days since the last purchase
    frequency = rng.poisson(3, n)                      # purchases in the last two years
    monetary = rng.gamma(2, 40, n)                     # average order value (EUR)
    age = rng.integers(18, 85, n)
    web = (rng.random(n) < 0.6).astype(int)            # buys online
    logit = (-3.4 - 0.01 * recency + 0.3 * frequency
             + 0.004 * monetary + 0.5 * web - 0.01 * (age - 45))
    X = np.c_[recency, frequency, monetary, age, web]
    # features, true response probability
    return X, 1 / (1 + np.exp(-logit)), rng

def pilot(n_pilot=10_000, seed=6):
    """Mail a random sample; split who responded into training and validation halves."""
    X, p, rng = customers(seed=seed)
    idx = rng.choice(len(p), n_pilot, replace=False)
    y = (rng.random(n_pilot) < p[idx]).astype(int)
    half = n_pilot // 2
    rest = np.setdiff1d(np.arange(len(p)), idx)        # the customers still to be mailed
    return X[idx[:half]], y[:half], X[idx[half:]], y[half:], X[rest], p[rest], rng
