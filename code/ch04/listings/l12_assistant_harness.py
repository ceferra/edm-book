import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(12)
n = 3000
df = pd.DataFrame({"income": np.exp(rng.normal(np.log(30), 0.5, n)).round(1),  # kEUR
                   "debt": rng.gamma(2, 5, n).round(1),                # thousand EUR
                   "age": rng.integers(18, 80, n),
                   "late_payments": rng.poisson(1, n),
                   "postcode": rng.integers(0, 1000, n)})              # no real effect
logit = -3.2 + 5.0 * df.debt / df.income + 0.5 * df.late_payments - 0.015 * (df.age - 45)
TARGET = "default"
df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

def cv_auc(data):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return cross_val_score(model, data.drop(columns=TARGET), data[TARGET], cv=5,
                           scoring="roc_auc").mean()

def run_safely(code, data):
    """Run proposed code on a copy of the data WITHOUT the target and enforce an
    append-only contract: original columns, values, rows and index must not change.
    (exec is not a security sandbox: real systems use isolated containers.)"""
    base = data.drop(columns=TARGET)
    env = {"df": base.copy(deep=True), "np": np, "pd": pd}
    try:
        exec(code, env)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    new = env.get("df")
    if not isinstance(new, pd.DataFrame) or not new.columns.is_unique:
        return None, "invalid result schema"
    if TARGET in new.columns:
        return None, "unexpected target column"
    if not new.index.equals(base.index):
        return None, "row index changed"
    if not base.columns.isin(new.columns).all():
        return None, "an original column was removed"
    if not new.loc[:, base.columns].equals(base):
        return None, "original rows or values changed"
    added = [c for c in new.columns if c not in base.columns]
    return (new[added], "ok") if added else (None, "no new column")

# What an assistant might answer to "propose a feature to predict default" (scripted
# here so that the result is reproducible; the second item is its answer after feedback)
PROPOSALS = [
    ['df["debt_to_income"] = df["debt"] / df["income"]'],
    ['df["utilisation"] = df["credit_used"] / df["credit_limit"]',
     'df["late_per_year"] = df["late_payments"] / (df["age"] - 17)'],
    ['df["postcode_risk"] = df.groupby("postcode")["default"].transform("mean")'],
    ['df = df[df["income"] > 15]\ndf["log_income"] = np.log(df["income"])'],
    ['df["age_squared"] = df["age"] ** 2'],
    # two silent errors that an index check alone would miss
    ['df = df.sort_values("income", ignore_index=True)\n'
     'df["ratio"] = df["debt"] / df["income"]'],
    ['df["income"] = df["income"] / 1000\ndf["income_k"] = df["income"]'],
]
data = df.copy()
print(f"baseline cross-validated AUC: {cv_auc(data):.3f}")
for attempts in PROPOSALS:
    for code in attempts:
        name = code.splitlines()[-1].split('"')[1]          # the column it creates
        feature, status = run_safely(code, data)
        if feature is None:
            verdict = "leakage: uses the target" if TARGET in status else status
            print(f"  {name:16s} rejected: {verdict}")
            if "KeyError" in status and TARGET not in status:
                print(f"    feedback to the assistant: {status}; available columns:")
                print(f"    {', '.join(c for c in data.columns if c != TARGET)}")
            continue
        gain = cv_auc(pd.concat([data, feature], axis=1)) - cv_auc(data)
        if gain >= 0.005:                    # a practical screening margin
            data = pd.concat([data, feature], axis=1)
        print(f"  {name:16s} {'accepted' if gain >= 0.005 else 'rejected'}: "
              f"AUC {gain:+.3f}")
        break
leaky = df.assign(postcode_risk=df.groupby("postcode")[TARGET].transform("mean"))
print(f"final AUC {cv_auc(data):.3f} with {', '.join(data.columns[6:])}")
print(f"had the leaky feature been accepted, the AUC would have been {cv_auc(leaky):.3f}")
