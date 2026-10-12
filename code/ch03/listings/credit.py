"""Synthetic credit-scoring data and four lending policies (Unit 3)."""
import numpy as np

def credit_data(n, seed=0):
    """Two groups of applicants; group b (A=1, 30%) is less creditworthy on average."""
    rng = np.random.default_rng(seed)
    A = (rng.random(n) < 0.3).astype(int)
    z = rng.normal(0, 1, n) - 0.8 * A                       # latent creditworthiness
    Y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * z + 1.2)))).astype(int)   # 1 = repays
    income = 2.0 * z + rng.normal(0, 1, n)
    debt = -1.0 * z + rng.normal(0, 1.2, n)
    zipcode = 1.5 * A + rng.normal(0, 0.6, n)               # neighbourhood: a proxy of A
    return np.c_[income, debt, zipcode], A, Y

GAIN, LOSS = 1.0, 4.0                    # profit of a repaid loan, loss of a default

def profit(R, A, Y, t):
    """Profit of accepting the applicants whose score reaches their group's threshold."""
    accept = R >= np.asarray(t)[A]
    return GAIN * np.sum(accept & (Y == 1)) - LOSS * np.sum(accept & (Y == 0))

def four_policies(R, A, Y):
    """Thresholds (group a, group b) of four lending policies, each maximising profit."""
    grid = np.quantile(R, np.linspace(0, 1, 401))           # candidate thresholds
    rates = np.linspace(0.005, 0.995, 199)                  # candidate common rates
    best = lambda cands: max(cands, key=lambda t: profit(R, A, Y, t))
    # threshold accepting a fraction r
    q = lambda mask, r: np.quantile(R[mask], 1 - r)

    # best threshold within one group
    def group_best(g):
        m = A == g
        return max(grid, key=lambda s: profit(R[m], np.zeros(m.sum(), int), Y[m], (s,)))

    return {"max profit": (group_best(0), group_best(1)),
            "single threshold": best([(s, s) for s in grid]),
            "independence": best([(q(A == 0, r), q(A == 1, r)) for r in rates]),
            "equal opportunity": best([(q((A == 0) & (Y == 1), r),
                                       q((A == 1) & (Y == 1), r)) for r in rates])}
