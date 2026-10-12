"""Synthetic apartment prices used in the explanation examples (Unit 3)."""
import numpy as np

FEATURES = ["park_nearby", "cat_allowed", "area_m2", "floor"]

def apartments(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.uniform(30, 120, n)
    park = (rng.random(n) < 0.5).astype(float)
    cat = (rng.random(n) < 0.6).astype(float)
    floor = rng.integers(0, 8, n).astype(float)
    price = (40 + 2.6 * area + 25 * park + 15 * cat + 3 * floor     # thousand EUR
             + 0.3 * park * (area - 75) + rng.normal(0, 12, n))
    return np.c_[park, cat, area, floor], price
