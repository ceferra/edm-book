import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

# (c) Data of Listing 2.9: ranking errors well is not enough, the window needs calibration
X, y = make_classification(n_samples=4000, n_features=20, n_informative=6, n_redundant=4,
                           flip_y=0.05, class_sep=0.8, random_state=5)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
for name, m in [("naive Bayes", GaussianNB()),
                ("naive Bayes + isotonic",
                 CalibratedClassifierCV(GaussianNB(), method="isotonic"))]:
    p1 = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    d = cautious_predict(np.c_[1 - p1, p1], K=(1 - t, t), w=w)
    loss = np.where(d == -1, c_r, np.where(d == 1, (y_te == 0) * c_fp, (y_te == 1) * c_fn))
    print(f"{name:23s} abstains {np.mean(d == -1):.3f}  mean cost {loss.mean():.3f}")
