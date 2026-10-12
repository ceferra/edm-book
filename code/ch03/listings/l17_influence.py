import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)
x = np.r_[rng.uniform(0, 5, 30), 12.0]                  # last point: high leverage...
y = np.r_[2 * x[:30] + 1 + rng.normal(0, 1.2, 30), 6.0] # ...and off the trend
X = x[:, None]
full = LinearRegression().fit(X, y)
yhat = full.predict(X)
s2 = np.sum((y - yhat) ** 2) / (len(y) - 2)              # residual variance (2 parameters)
dfbeta, cook = [], []
for i in range(len(y)):                       # deletion diagnostics: refit without i
    m = LinearRegression().fit(np.delete(X, i, 0), np.delete(y, i))
    dfbeta.append(full.coef_[0] - m.coef_[0])
    cook.append(np.sum((yhat - m.predict(X)) ** 2) / (2 * s2))
for i in np.argsort(cook)[::-1][:3]:
    print(f"instance {i:2d}: x={x[i]:5.2f}  DFBETA={dfbeta[i]:+.3f}  "
          f"Cook's D={cook[i]:.3f}")
