import numpy as np

# (a) Costs: rows = decision (a, b, c, unknown), columns = actual class (a, b, c)
C = np.array([[-2.5,  4.0,  2.0],
              [ 2.1, -3.5,  0.0],
              [ 1.2,  1.3, -4.0],
              [ 0.0,  0.0,  0.0]])
decisions = np.array(["a", "b", "c", "unknown"])
for p in ([0.40, 0.35, 0.25], [0.50, 0.28, 0.22]):
    ec = C @ np.array(p)                       # expected cost of each decision
    print(f"p={p}  expected costs={ec.round(3)}  ->  {decisions[ec.argmin()]}")

# (b) Two classes: the window rule with a cost-matched K is the optimal reject rule
def cautious_predict(P, K, w):
    tau = (1 - np.asarray(K)) * w + np.asarray(K)      # one threshold per class
    ok = P >= tau
    pred = np.where(ok, P / tau, -np.inf).argmax(axis=1)
    pred[~ok.any(axis=1)] = -1                          # -1 means "unknown"
    return pred

c_fp, c_fn, c_r = 1.0, 4.0, 0.5
t = c_fp / (c_fp + c_fn)                                # cost threshold t*
w = 1 - c_r * (c_fp + c_fn) / (c_fp * c_fn)
p = np.linspace(0.0005, 0.9995, 1000)                   # P(positive)
window = cautious_predict(np.c_[1 - p, p], K=(1 - t, t), w=w)
cost = np.c_[p * c_fn, (1 - p) * c_fp, np.full_like(p, c_r)]   # predict 0, 1, abstain
bayes = np.array([0, 1, -1])[cost.argmin(axis=1)]
print(f"t*={t:.2f}  w={w:.3f}  identical decisions: {np.mean(window == bayes):.3f}")
print(f"abstains for p in [{p[window == -1].min():.4f}, {p[window == -1].max():.4f}]")
