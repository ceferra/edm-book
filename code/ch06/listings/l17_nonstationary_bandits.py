import numpy as np

rng = np.random.default_rng(17)
T, SWITCH = 40_000, 20_000
before = np.array([0.050, 0.055, 0.060])
after = np.array([0.060, 0.055, 0.045])              # the best variant becomes the worst

def run(discount):
    """Thompson sampling; with discount < 1, old evidence fades away."""
    clicks, shows, regret, lost = np.zeros(3), np.zeros(3), np.empty(T), 0.0
    for t in range(T):
        ctr = before if t < SWITCH else after
        arm = int(np.argmax(rng.beta(1 + clicks, 1 + shows - clicks)))
        clicks *= discount
        shows *= discount
        clicks[arm] += rng.random() < ctr[arm]
        shows[arm] += 1
        lost += ctr.max() - ctr[arm]
        regret[t] = lost
    return regret

curves = {}
for name, discount in [("Thompson", 1.0), ("discounted Thompson", 0.999)]:
    curves[name] = np.mean([run(discount) for _ in range(10)], axis=0)
    c = curves[name]
    print(f"{name:20s} clicks lost: {c[SWITCH - 1]:5.1f} before the change, "
          f"{c[-1] - c[SWITCH - 1]:6.1f} after it")
