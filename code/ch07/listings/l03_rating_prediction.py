import numpy as np
from recsys_models import fit_biases, fit_mf, knn, make_ratings

R, test = make_ratings()
u, i, r = test[:, 0].astype(int), test[:, 1].astype(int), test[:, 2]
print(f"{R.shape[0]} users, {R.shape[1]} items, {np.sum(~np.isnan(R))} training ratings "
      f"({np.mean(~np.isnan(R)):.1%} of the matrix), {len(r)} test ratings")
rmse = lambda pred: np.sqrt(np.mean((np.asarray(pred) - r) ** 2))
models = {"global mean": lambda u, i: np.full(len(u), np.nanmean(R)),
          "user and item biases": fit_biases(R),
          "user-based neighbours": np.vectorize(knn(R, by="user")),
          "item-based neighbours": np.vectorize(knn(R, by="item")),
          "matrix factorisation": fit_mf(R)}
results = {name: rmse(model(u, i)) for name, model in models.items()}
for name, value in results.items():
    print(f"{name:22s} RMSE {value:.3f}")
