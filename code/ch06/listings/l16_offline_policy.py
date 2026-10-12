import numpy as np

rng = np.random.default_rng(16)
# rows: user segment, columns: article
ctr = np.array([[0.05, 0.04, 0.03],
                [0.09, 0.05, 0.04],
                [0.03, 0.04, 0.07]])
segment_share = np.array([0.5, 0.3, 0.2])
logging = np.array([[0.6, 0.2, 0.2],                 # the current, randomised policy
                    [0.6, 0.2, 0.2],
                    [0.2, 0.2, 0.6]])
n = 200_000
s = rng.choice(3, n, p=segment_share)
a = (rng.random(n)[:, None] > np.cumsum(logging[s], axis=1)).sum(axis=1)
r = (rng.random(n) < ctr[s, a]).astype(float)         # the logs: segment, article, click
new = np.array([1, 0, 2])                            # the policy we want to evaluate

truth = np.sum(segment_share * ctr[np.arange(3), new])
match = a == new[s]
naive = r[match].mean()                              # logged events where both agree
terms = match * r / logging[s, a]                    # inverse propensity scoring
ips, se = terms.mean(), terms.std() / np.sqrt(n)
print(f"CTR of the logging policy: {r.mean():.4f}")
print(f"new policy: true CTR {truth:.4f}, naive estimate {naive:.4f}, "
      f"IPS {ips:.4f} +/- {1.96 * se:.4f}")
print(f"the estimates use the {match.sum():,} logged events in which the policies agree")
