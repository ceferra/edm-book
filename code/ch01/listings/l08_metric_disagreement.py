import numpy as np
from sklearn.metrics import roc_auc_score

mae = lambda y, p: np.mean(np.abs(y - p))
mse = lambda y, p: np.mean((y - p) ** 2)
acc = lambda y, p: np.mean((p >= 0.5) == y)

y = np.array([1, 1])                                   # (a) MAE vs. MSE
m1, m2 = np.array([1.0, 0.2]), np.array([0.55, 0.55])
print(f"(a) MAE: {mae(y, m1):.3f} vs {mae(y, m2):.3f} | "
      f"MSE: {mse(y, m1):.3f} vs {mse(y, m2):.3f}")
y = np.array([1, 1, 0, 0])                             # (b) accuracy vs. MAE
m1, m2 = np.array([.51, .51, .49, .49]), np.array([.99, .99, .01, .60])
print(f"(b) Acc: {acc(y, m1):.2f} vs {acc(y, m2):.2f} | "
      f"MAE: {mae(y, m1):.3f} vs {mae(y, m2):.3f}")
m1, m2 = np.array([.90, .40, .30, .45]), np.array([.90, .80, .70, .60])
print(f"(c) Acc: {acc(y, m1):.2f} vs {acc(y, m2):.2f} | "    # (c) accuracy vs. AUC
      f"AUC: {roc_auc_score(y, m1):.2f} vs {roc_auc_score(y, m2):.2f}")
