import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(1)
n = 100_000
coef = lambda cols, y: LinearRegression().fit(np.column_stack(cols), y).coef_[0]
noise = lambda: rng.normal(size=n)

# (a) fork: Z is a common cause of X and Y; the effect of X on Y is 1
Z = noise()
X = Z + noise()
Y = X + 2 * Z + noise()
print(f"fork:     naive {coef([X], Y):.2f}   adjusting for Z {coef([X, Z], Y):.2f}")
# (b) chain: X -> M -> Y plus a direct effect; total 1 + 2 * 1.5 = 4, direct 1
X = noise()
M = 2 * X + noise()
Y = X + 1.5 * M + noise()
print(f"mediator: naive {coef([X], Y):.2f}   adjusting for M {coef([X, M], Y):.2f}")
# (c) collider: X -> C <- Y; X has no effect on Y
X = noise()
Y = noise()
C = X + Y + noise()
print(f"collider: naive {coef([X], Y):.2f}   adjusting for C {coef([X, C], Y):.2f}")
