import numpy as np
from scipy.stats import fisher_exact

rng = np.random.default_rng(11)
STAGES = [0.01, 0.05, 0.20, 0.50, 0.95]   # share of requests for the new model;
DAILY = 100_000                           # 5% stay on the old one as a control
BASE = {"conversion": 0.050, "errors": 0.001}        # the current model

def worse(events_new, n_new, events_old, n_old, higher_is_better, alpha=0.01):
    """Exact one-sided test of harm (Fisher), suitable for rare events such as errors."""
    table = [[events_new, n_new - events_new], [events_old, n_old - events_old]]
    side = "less" if higher_is_better else "greater"
    return fisher_exact(table, alternative=side).pvalue < alpha

def rollout(model):
    exposed = 0
    for share in STAGES:                             # one day of requests per stage
        n_new = int(DAILY * share)
        n_old = DAILY - n_new
        exposed += n_new
        for metric, better in [("errors", False), ("conversion", True)]:   # guardrails
            e_new = rng.binomial(n_new, model[metric])
            e_old = rng.binomial(n_old, BASE[metric])
            if worse(e_new, n_new, e_old, n_old, better):
                return (f"rolled back at {share:.0%} ({metric}), "
                        f"{exposed:,} requests served")
    return f"passed every stage, {exposed:,} requests served during the rollout"

for name, model in [("good model", {"conversion": 0.052, "errors": 0.001}),
                    ("crashes more", {"conversion": 0.050, "errors": 0.020}),
                    ("converts less", {"conversion": 0.046, "errors": 0.001})]:
    print(f"{name:14s} {rollout(model)}")
false = np.mean(["rolled" in rollout(BASE) for _ in range(2000)])
print(f"an identical model is rolled back in {false:.1%} of 2000 rollouts "
      "(10 tests at the 1% level)")
