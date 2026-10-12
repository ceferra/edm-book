import numpy as np
from sklearn.metrics import classification_report

cm = np.array([[8, 2, 0],      # rows: actual class (cat, dog, rabbit)
               [1, 4, 1],      # columns: predicted class
               [0, 1, 3]])
y_true = np.repeat(np.arange(3), cm.sum(axis=1))
y_pred = np.concatenate([np.repeat(np.arange(3), row) for row in cm])
print(classification_report(y_true, y_pred, digits=3,
                            target_names=["cat", "dog", "rabbit"]))
