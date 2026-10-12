import time
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(1)
n = 5000
X = pd.DataFrame({"income": rng.normal(30, 8, n).astype(np.float32),
                  "debt_ratio": rng.uniform(0, 0.8, n).astype(np.float32),
                  "contract": rng.choice(["permanent", "temporary", "self-employed"], n),
                  "region": rng.choice(["north", "south", "east", "west"], n)})
logit = (0.08 * (X.income - 30) - 3 * (X.debt_ratio - 0.4)
         + 0.8 * (X.contract == "permanent") - 0.4)
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
prep = ColumnTransformer([("num", StandardScaler(), ["income", "debt_ratio"]),
                          ("cat", OneHotEncoder(handle_unknown="ignore"),
                           ["contract", "region"])])
pipe = Pipeline([("prep", prep), ("model", GradientBoostingClassifier(random_state=0))])
pipe.fit(X, y)

onx = to_onnx(pipe, X[:1], options={GradientBoostingClassifier: {"zipmap": False}})
print("inputs:", [i.name for i in onx.graph.input])
print("opsets:", {d.domain or "ai.onnx": d.version for d in onx.opset_import})
print("operators:", ", ".join(sorted({node.op_type for node in onx.graph.node})))
session = ort.InferenceSession(onx.SerializeToString(),
                               providers=["CPUExecutionProvider"])
feeds = lambda df: {c: df[[c]].to_numpy() for c in df.columns}   # one input per column
labels, probs = session.run(None, feeds(X))
gap = np.abs(probs[:, 1] - pipe.predict_proba(X)[:, 1]).max()
print(f"same labels: {np.array_equal(labels, pipe.predict(X))}, "
      f"largest probability difference: {gap:.1e}")

def median_ms(f, reps):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return 1000 * np.median(times)

one, one_feeds, all_feeds = X[:1], feeds(X[:1]), feeds(X)   # inputs prepared once
sk_one = median_ms(lambda: pipe.predict_proba(one), 300)
ort_one = median_ms(lambda: session.run(None, one_feeds), 300)
print(f"one request: scikit-learn {sk_one:.2f} ms, ONNX Runtime {ort_one:.3f} ms")
sk_all = median_ms(lambda: pipe.predict_proba(X), 10)
ort_all = median_ms(lambda: session.run(None, all_feeds), 10)
print(f"{n} rows:   scikit-learn {sk_all:.1f} ms, ONNX Runtime {ort_all:.1f} ms")
