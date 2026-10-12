import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(2)
T = 6000
X = rng.normal(size=(T, 2))
t = np.arange(T)
angle = np.where(t < 2000, 0.0, np.pi / 2) + np.clip((t - 4000) / 1000, 0, 1) * np.pi / 3
y = ((X * np.c_[np.cos(angle), np.sin(angle)]).sum(axis=1) > 0).astype(int)
y = np.where(rng.random(T) < 0.1, 1 - y, y)       # 10% label noise
# the concept changes suddenly at t = 2000 and incrementally between t = 4000 and 5000

class DDM:
    """Drift detection method (Gama et al., 2004) on a stream of 0/1 errors."""
    def __init__(self, warning=2.0, drift=3.0, min_n=100):
        self.warning, self.drift, self.min_n = warning, drift, min_n
        self.reset()

    def reset(self):
        self.n, self.errors, self.best, self.p_min, self.s_min = 0, 0, np.inf, 0, 0
        self.warning_at = None

    def update(self, error, t):
        """Returns None, or the estimated start of the new concept when drift is found."""
        self.n += 1
        self.errors += error
        p = self.errors / self.n
        s = np.sqrt(p * (1 - p) / self.n)
        if self.n < self.min_n:
            return None
        if p + s < self.best:                       # the best moment seen so far
            self.best, self.p_min, self.s_min = p + s, p, s
        if p + s > self.p_min + self.drift * self.s_min:      # strict: errors needed
            return t if self.warning_at is None else self.warning_at
        if p + s > self.p_min + self.warning * self.s_min:
            self.warning_at = t if self.warning_at is None else self.warning_at
        else:
            self.warning_at = None                  # a false warning
        return None

class PageHinkley:
    """Page-Hinkley test for an increase in the mean of a stream."""
    def __init__(self, delta=0.01, threshold=20):
        self.delta, self.threshold = delta, threshold
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.m_min, self.t_min = 0, 0.0, 0.0, 0.0, None

    def update(self, x, t):
        """Returns None, or the estimated change point when the test fires."""
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m += x - self.mean - self.delta
        if self.t_min is None or self.m < self.m_min:
            self.m_min, self.t_min = self.m, t
        return self.t_min if self.m - self.m_min > self.threshold else None

def run(detector=None, window=None, start=500, refresh=50, cap=2000):
    """Prequential evaluation: predict each example, then learn its label. Every `refresh`
    examples the model is retrained on the last `window` examples or, with a detector,
    on the data since the last detected change (an adaptive window)."""
    model = LogisticRegression().fit(X[:start], y[:start])
    since, errors, alarms = 0, [], []
    for i in range(start, T):
        e = int(model.predict(X[i:i + 1])[0] != y[i])
        errors.append(e)
        refit = (i - start + 1) % refresh == 0
        if detector is not None:
            change = detector.update(e, i)
            if change is not None:
                alarms.append(i)
                detector.reset()
                since, refit = max(change, i - 100), True   # noisy estimate: bound it
        if refit and (window is not None or detector is not None):
            first = i + 1 - window if window else max(since, i + 1 - cap)
            first = min(first, i + 1 - 30)           # at least 30 examples
            if np.unique(y[first:i + 1]).size == 2:    # both classes needed to refit
                model = LogisticRegression().fit(X[first:i + 1], y[first:i + 1])
    return np.array(errors), alarms

results = {"static model": run(), "sliding window": run(window=300),
           "DDM": run(DDM()), "Page-Hinkley": run(PageHinkley())}
print("                  overall   t < 2000   2000-4000   4000-6000   alarms")
for name, (errors, alarms) in results.items():
    part = [1 - errors[a - 500:b - 500].mean() for a, b in [(500, 2000), (2000, 4000),
                                                           (4000, T)]]
    print(f"{name:18s} {1 - errors.mean():.3f}     {part[0]:.3f}      {part[1]:.3f}"
          f"       {part[2]:.3f}    {', '.join(map(str, alarms)) or '-'}")
