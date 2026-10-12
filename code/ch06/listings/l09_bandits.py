import numpy as np

rng = np.random.default_rng(8)
ctr = np.array([0.050, 0.055, 0.060])              # true click rates, unknown to us
T = 50_000

def ab_test(t, clicks, shows, explore=15_000):     # equal split, then the winner
    return t % 3 if t < explore else int(np.argmax(clicks / shows))

def epsilon_greedy(t, clicks, shows, eps=0.1):     # usually the best so far
    if shows.min() == 0 or rng.random() < eps:
        return int(rng.integers(3))
    return int(np.argmax(clicks / shows))

def thompson(t, clicks, shows):                    # a plausible CTR per variant
    return int(np.argmax(rng.beta(1 + clicks, 1 + shows - clicks)))

# optimism: mean plus an uncertainty bonus
def ucb(t, clicks, shows):
    if shows.min() == 0:
        return int(np.argmin(shows))
    return int(np.argmax(clicks / shows + np.sqrt(2 * np.log(t + 1) / shows)))

def run(policy):
    clicks, shows, regret = np.zeros(3), np.zeros(3), np.empty(T)
    lost = 0.0
    for t in range(T):
        arm = policy(t, clicks, shows)
        clicks[arm] += rng.random() < ctr[arm]
        shows[arm] += 1
        lost += ctr.max() - ctr[arm]               # expected clicks lost by this choice
        regret[t] = lost
    return regret, shows[2] / T

curves = {}
for name, policy in [("A/B test, then winner", ab_test),
                     ("epsilon-greedy", epsilon_greedy), ("Thompson sampling", thompson),
                     ("UCB1", ucb)]:
    runs = [run(policy) for _ in range(10)]
    curves[name] = np.mean([r[0] for r in runs], axis=0)      # mean regret over time
    print(f"{name:22s} clicks lost {np.mean([r[0][-1] for r in runs]):6.1f}   "
          f"traffic to the best variant {np.mean([r[1] for r in runs]):.2f}")
