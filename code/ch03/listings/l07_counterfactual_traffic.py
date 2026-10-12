import numpy as np

rng = np.random.default_rng(2)
n = 2_000_000
# bad day, accident on route 0 / 1, route choice
U, U0, U1, UX = rng.random((4, n)) < 0.5

def morning(X, U, U0, U1):
    """Y = 1 means a slow morning: a bad day or an accident on the chosen route."""
    return np.where(X == 0, U | U0, U | U1)

X = UX.astype(int)
Y = morning(X, U, U0, U1)
p_do = np.mean(morning(0, U, U0, U1) == 0)
print(f"P(Y=0 | do(X:=0))          = {p_do:.3f}  (exact 1/4)")
E = (X == 1) & (Y == 1)                                  # evidence: route 1, slow morning
print(f"P(U=1 | X=1, Y=1)          = {U[E].mean():.3f}  (exact 2/3)")
cf = morning(0, U[E], U0[E], U1[E])                      # abduction, action, prediction
print(f"P(Y_(X:=0) = 0 | X=1, Y=1) = {np.mean(cf == 0):.3f}  (exact 1/6)")
