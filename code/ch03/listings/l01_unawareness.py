import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from credit import credit_data

# X = (income, debt, zipcode); A is not in X
X, A, Y = credit_data(40000)
pA = cross_val_predict(LogisticRegression(), X, A, cv=5, method="predict_proba")[:, 1]
print(f"AUC of predicting the group from the features: {roc_auc_score(A, pA):.3f}")

X_tr, X_te, A_tr, A_te, Y_tr, Y_te = train_test_split(X, A, Y, test_size=0.5,
                                                      random_state=0)
for cols, name in [([0, 1, 2], "income, debt, zipcode"), ([0, 1], "income, debt")]:
    m = LogisticRegression().fit(X_tr[:, cols], Y_tr)
    accept = m.predict_proba(X_te[:, cols])[:, 1] >= 0.8         # lend if P(repay) >= 0.8
    print(f"{name:21s} weights={np.round(m.coef_[0], 2)}  "
          f"acceptance a={accept[A_te == 0].mean():.2f} b={accept[A_te == 1].mean():.2f}")
print(f"repayment rate: a={Y[A == 0].mean():.2f}  b={Y[A == 1].mean():.2f}")
