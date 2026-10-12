import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             median_absolute_error, mean_absolute_percentage_error,
                             explained_variance_score)

y_pred = np.array([100, 102, 105, 95, 101, 105, 105, 40, 220, 100])  # million EUR
y_true = np.array([102, 110, 95, 75, 103, 110, 98, 32, 215, 103])
rae = np.abs(y_true - y_pred).sum() / np.abs(y_true - y_true.mean()).sum()
print(f"MAE  = {mean_absolute_error(y_true, y_pred):.3f}")
print(f"MSE  = {mean_squared_error(y_true, y_pred):.3f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
print(f"MedAE= {median_absolute_error(y_true, y_pred):.3f}")
print(f"MAPE = {mean_absolute_percentage_error(y_true, y_pred):.4f}")
print(f"RAE  = {rae:.4f}")
print(f"R2   = {r2_score(y_true, y_pred):.4f}")
print(f"EV   = {explained_variance_score(y_true, y_pred):.4f}")
print(f"bias = {np.mean(y_pred - y_true):.3f}   (mean of yhat - y)")
