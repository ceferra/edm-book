import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             precision_score, f1_score, balanced_accuracy_score,
                             matthews_corrcoef, cohen_kappa_score)

# Purchase example: 1 = "buy" (positive class), 0 = "no buy"
y_true = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0])

# scikit-learn convention: rows = ACTUAL class, columns = PREDICTED class
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
print(f"accuracy          {accuracy_score(y_true, y_pred):.4f}")
print(f"sensitivity (TPR) {recall_score(y_true, y_pred):.4f}")
print(f"specificity (TNR) {recall_score(y_true, y_pred, pos_label=0):.4f}")
print(f"precision (PPV)   {precision_score(y_true, y_pred):.4f}")
print(f"F1 score          {f1_score(y_true, y_pred):.4f}")
print(f"balanced accuracy {balanced_accuracy_score(y_true, y_pred):.4f}")
print(f"MCC               {matthews_corrcoef(y_true, y_pred):.4f}")
print(f"Cohen's kappa     {cohen_kappa_score(y_true, y_pred):.4f}")
