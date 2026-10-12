import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(10)

def applicants(n):
    income = rng.normal(30, 8, n)                    # thousand EUR
    debt, age = rng.uniform(0, 0.8, n), rng.integers(18, 80, n)
    logit = 0.12 * (income - 30) - 4 * (debt - 0.4) + 0.03 * (age - 45)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return np.c_[income, debt, age], y

X_train, y_train = applicants(5000)
X_test, y_test = applicants(5000)
# in production
current = LogisticRegression(max_iter=1000).fit(X_train[:, :2], y_train)
candidate = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print(f"offline accuracy: current {current.score(X_test[:, :2], y_test):.3f}, "
      f"candidate {candidate.score(X_test, y_test):.3f}")

X_live, y_live = applicants(20000)                   # a week of real requests
served = current.predict(X_live[:, :2])              # what the users receive
X_feed = X_live.copy()
# bug: the new feature service sends EUR
X_feed[:, 0] *= 1000
shadow = candidate.predict(X_feed)                   # computed and logged, never used
print(f"shadow week: agreement {np.mean(served == shadow):.3f}, approval rate "
      f"current {served.mean():.3f}, candidate {shadow.mean():.3f}")
print(f"once the outcomes arrive: current {np.mean(served == y_live):.3f}, "
      f"candidate {np.mean(shadow == y_live):.3f}")
fixed = candidate.predict(X_live)                    # after fixing the units
print(f"after the fix: agreement {np.mean(served == fixed):.3f}, "
      f"candidate accuracy {np.mean(fixed == y_live):.3f}")
