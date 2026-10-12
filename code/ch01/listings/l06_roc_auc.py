import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

y = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0])                     # actual class
s = np.array([.95, .90, .85, .80, .70, .70, .60, .55, .40, .20])  # scores
fpr, tpr, thr = roc_curve(y, s)
print("FPR:", fpr.round(2))
print("TPR:", tpr.round(2))
print("AUC (scikit-learn):", roc_auc_score(y, s))

# AUC as the Mann-Whitney-Wilcoxon statistic: compare all (pos, neg) pairs
pos, neg = s[y == 1], s[y == 0]
diff = pos[:, None] - neg[None, :]
print("AUC (pairs):       ", ((diff > 0) + 0.5 * (diff == 0)).mean())
