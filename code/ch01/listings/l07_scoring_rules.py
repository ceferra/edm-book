import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, log_loss

y = np.array([1, 0, 1, 1, 0])
for p3 in (0.60, 0.68):                       # probability given to the third example
    p = np.array([0.80, 0.70, p3, 0.55, 0.40])
    print(f"p3={p3:.2f}  AUC={roc_auc_score(y, p):.3f}  "
          f"Acc={accuracy_score(y, p >= 0.5):.2f}  "
          f"Brier={brier_score_loss(y, p):.4f}  LogLoss={log_loss(y, p):.4f}")
