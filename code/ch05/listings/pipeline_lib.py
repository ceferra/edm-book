"""A small credit-scoring pipeline, written so that it can be tested (Unit 5)."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

FEATURES = ["age", "income", "debt"]

def clip_age(df):
    """A fixed rule, the same for every batch: keep the features, clip the ages."""
    out = df[FEATURES].copy()
    out["age"] = out["age"].clip(18, 100)
    return out

def validate(df):
    """Data quality checks on schema, values and ranges; returns the problems found."""
    problems = [f"missing column {c}" for c in FEATURES if c not in df]
    if problems:
        return problems
    X = df[FEATURES]
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in FEATURES):
        return ["non-numeric values"]
    if np.isinf(X.to_numpy(dtype=float)).any():
        problems.append("infinite values")
    if (X["income"] < 0).any():
        problems.append("negative income")
    if (X["debt"] < 0).any():
        problems.append("negative debt")
    return problems

def train(df, y):
    """Fixed cleaning, a median imputation learned on the training data only and the
    classifier, in one pipeline saved and reused unchanged for every prediction."""
    model = make_pipeline(FunctionTransformer(clip_age),
                          SimpleImputer(strategy="median", keep_empty_features=True),
                          LogisticRegression(max_iter=1000))
    return model.fit(df, y)

def make_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"name": [f"client{i}" for i in range(n)],
                       "age": rng.integers(18, 80, n), "income": rng.normal(30, 8, n),
                       "debt": rng.uniform(0, 20, n)})
    logit = 0.15 * (df.income - 30) - 0.2 * (df.debt - 10)
    return df, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
