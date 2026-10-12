import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from credit import credit_data, four_policies, profit

X, A, Y = credit_data(40000)
X_tr, X_te, A_tr, A_te, Y_tr, Y_te = train_test_split(X, A, Y, test_size=0.5,
                                                      random_state=0)
X_va, X_te, A_va, A_te, Y_va, Y_te = train_test_split(X_te, A_te, Y_te, test_size=0.5,
                                                      random_state=1)
model = LogisticRegression().fit(X_tr, Y_tr)
R_va, R = model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1]
policies = four_policies(R_va, A_va, Y_va)       # thresholds chosen on validation
# ... and evaluated, held fixed, on the test data
top = profit(R, A_te, Y_te, policies["max profit"])
print("test data         thresholds  acceptance    TPR         FPR      profit")
for name, t in policies.items():
    acc = R >= np.asarray(t)[A_te]
    rate = lambda m: f"{acc[m & (A_te == 0)].mean():.2f} {acc[m & (A_te == 1)].mean():.2f}"
    print(f"{name:17s} {t[0]:.2f} {t[1]:.2f}   {rate(Y_te >= 0)}    {rate(Y_te == 1)}   "
          f"{rate(Y_te == 0)}   {profit(R, A_te, Y_te, t) / top:.1%}")
