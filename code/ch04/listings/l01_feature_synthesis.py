import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
n = 2000
customers = pd.DataFrame({"customer": np.arange(n), "age": rng.integers(18, 80, n),
                          "premium": (rng.random(n) < 0.3).astype(int)})
# latent variables (never observed): engagement and how much activity faded
engagement, fading = rng.gamma(2.0, 1.0, n), rng.random(n)
k = rng.poisson(3 + 6 * engagement)                        # transactions per customer
owner = np.repeat(np.arange(n), k)
tx = pd.DataFrame({"customer": owner,
                   "amount": rng.gamma(2.0, 15.0, k.sum()) * (0.5 + engagement[owner]),
                   "days_ago": rng.uniform(300 * fading[owner] ** 2, 365)})
logit = -1.0 + 4.0 * fading ** 2 - 0.6 * engagement
churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

def synthesize(parent, child, key, columns, primitives):
    """Depth-1 deep feature synthesis: every primitive applied to every child column."""
    agg = child.groupby(key)[columns].agg(primitives)
    agg.columns = [f"{p.upper()}(tx.{c})" for c, p in agg.columns]
    return parent.merge(agg, left_on=key, right_index=True, how="left")

data = synthesize(customers, tx, "customer", ["amount", "days_ago"],
                  ["count", "sum", "mean", "max", "min", "std"])
recent = tx[tx.days_ago < 30].groupby("customer").size()       # a "where" primitive
data["COUNT(tx WHERE days_ago < 30)"] = data.customer.map(recent).fillna(0)
data = data.fillna(-1)
new = [c for c in data.columns if "(" in c]
print(f"{len(new)} synthesized features, e.g. {new[2]}, {new[10]}")
model = GradientBoostingClassifier(random_state=0)
for name, cols in [("customer table only", ["age", "premium"]),
                   ("with synthesized features", ["age", "premium"] + new)]:
    auc = cross_val_score(model, data[cols], churn, cv=5, scoring="roc_auc").mean()
    print(f"{name:26s} AUC = {auc:.3f}")
