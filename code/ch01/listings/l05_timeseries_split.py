import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.arange(12).reshape(-1, 1)          # 12 consecutive time steps
for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=4, test_size=2).split(X), 1):
    print(f"fold {k}: train={tr.tolist()}  test={te.tolist()}")
