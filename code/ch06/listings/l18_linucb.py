import numpy as np

rng = np.random.default_rng(18)
d, K, T = 5, 3, 20_000
# each variant appeals to different users
theta = rng.normal(0, 1, (K, d))

def ctr(x):
    """Click probability of each variant for a user with features x."""
    return 1 / (1 + np.exp(-(0.6 * theta @ x - 3.0)))

class LinUCB:
    """Li et al. (2010): a ridge regression per variant plus an optimism bonus."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.A_inv = np.stack([np.eye(d + 1)] * K)
        self.b = np.zeros((K, d + 1))

    def choose(self, x):
        z = np.r_[x, 1.0]
        means = np.einsum("kij,kj,i->k", self.A_inv, self.b, z)
        bonus = np.sqrt(np.einsum("i,kij,j->k", z, self.A_inv, z))
        return int(np.argmax(means + self.alpha * bonus))

    def update(self, arm, x, reward):
        z = np.r_[x, 1.0]
        Az = self.A_inv[arm] @ z                       # Sherman-Morrison update
        self.A_inv[arm] -= np.outer(Az, Az) / (1 + z @ Az)
        self.b[arm] += reward * z

def simulate(policy):
    clicks, shows, lin, lost = np.zeros(K), np.zeros(K), LinUCB(), 0.0
    regret = np.empty(T)
    for t in range(T):
        x = rng.normal(size=d)
        p = ctr(x)
        if policy == "LinUCB":
            arm = lin.choose(x)
        elif policy == "Thompson (no context)":
            arm = int(np.argmax(rng.beta(1 + clicks, 1 + shows - clicks)))
        else:
            arm = int(np.argmax(p))                    # oracle: knows the click rates
        reward = float(rng.random() < p[arm])
        clicks[arm] += reward
        shows[arm] += 1
        lin.update(arm, x, reward)
        lost += p.max() - p[arm]
        regret[t] = lost
    return regret, clicks.sum() / T

results = {}
for policy in ["oracle", "Thompson (no context)", "LinUCB"]:
    regret, rate = simulate(policy)
    results[policy] = regret
    print(f"{policy:22s} CTR {rate:.4f}   expected clicks lost {regret[-1]:7.1f}")
