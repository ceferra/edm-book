"""Synthetic loan applications used in the interpretability examples (Unit 3)."""
import numpy as np

FEATURES = ["income", "debt_ratio", "years_job", "prev_default"]

def loan_data(n=5000, seed=3):
    rng = np.random.default_rng(seed)
    income = rng.normal(30, 8, n)                       # thousand EUR per year
    debt = rng.uniform(0, 0.8, n)                       # debt-to-income ratio
    years = rng.integers(0, 30, n).astype(float)        # years in the current job
    prev = (rng.random(n) < 0.15).astype(float)         # previous default (0/1)
    logit = -2 + 0.08 * income - 3 * debt + 0.03 * years - 1.2 * prev
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)     # 1 = repays
    return np.c_[income, debt, years, prev], y
