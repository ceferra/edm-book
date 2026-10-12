import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from credit import credit_data

X, A, Y = credit_data(40000)
X_tr, X_te, A_tr, A_te, Y_tr, Y_te = train_test_split(X, A, Y, test_size=0.5,
                                                      random_state=0)
R = LogisticRegression().fit(X_tr, Y_tr).predict_proba(X_te)[:, 1]   # score (A not used)
D = R >= 0.8                                                          # decision: lend
for g, name in [(0, "a"), (1, "b")]:
    m = A_te == g
    gap = R[m].mean() - Y_te[m].mean()  # calibration-in-the-large
    print(f"group {name}: base={Y_te[m].mean():.2f} accept={D[m].mean():.2f} "
          f"TPR={D[m & (Y_te == 1)].mean():.2f} FPR={D[m & (Y_te == 0)].mean():.2f} "
          f"score-base={gap:+.3f}")
    print(f"         PPV={Y_te[m & D].mean():.2f} NPV={1 - Y_te[m & ~D].mean():.2f}"
          f"  (predictive values of the decision)")
print(f"demographic parity ratio b/a = {D[A_te == 1].mean() / D[A_te == 0].mean():.2f}")
