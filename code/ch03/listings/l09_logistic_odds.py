import numpy as np
from sklearn.linear_model import LogisticRegression
from loans import FEATURES, loan_data

X, y = loan_data()
m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)      # practically unregularised
for name, b in zip(FEATURES, m.coef_[0]):
    print(f"{name:13s} beta={b:7.3f}   odds ratio exp(beta)={np.exp(b):6.3f}")
odds = 0.3 / 0.7 * np.exp(10 * m.coef_[0][0])               # P(repay)=0.3, +10k EUR income
p_new = odds / (1 + odds)
print(f"an applicant with P(repay)=0.30 earning 10k EUR more: P(repay)={p_new:.2f}")
