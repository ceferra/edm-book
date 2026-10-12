import pickle
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
kb = lambda obj: len(pickle.dumps(obj)) / 1024             # what we ship to the device
results = []
for n in (300, 30):
    rf = RandomForestClassifier(n, random_state=0).fit(X_tr, y_tr)
    results.append((f"random forest, {n} trees", kb(rf), rf.score(X_te, y_te)))
tree = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X_tr, y_tr)
results.append(("decision tree, depth 10", kb(tree), tree.score(X_te, y_te)))

mlp = MLPClassifier((64,), max_iter=1000, random_state=0).fit(X_tr, y_tr)
def mlp_accuracy(W, b):                              # inference with given weights
    h = np.maximum(X_te @ W[0].astype(float) + b[0], 0)
    return np.mean(np.argmax(h @ W[1].astype(float) + b[1], axis=1) == y_te)
b = [v.astype(np.float32) for v in mlp.intercepts_]
for dtype in (np.float32, np.float16):               # lower precision
    W = [w.astype(dtype) for w in mlp.coefs_]
    results.append((f"neural network, {np.dtype(dtype).name}", kb((W, b)),
                    mlp_accuracy(W, b)))
scales = [np.abs(w).max() / 127 for w in mlp.coefs_]  # int8: one scale per layer
Wq = [np.round(w / s).astype(np.int8) for w, s in zip(mlp.coefs_, scales)]
results.append(("neural network, int8", kb((Wq, scales, b)),
                mlp_accuracy([q * s for q, s in zip(Wq, scales)], b)))

top = np.argsort(-rf.feature_importances_)[:16]        # fewer features: 16 pixels
lr = LogisticRegression(max_iter=2000).fit(X_tr[:, top], y_tr)
W16, b16 = lr.coef_.astype(np.float16), lr.intercept_.astype(np.float16)
scores = X_te[:, top] @ W16.astype(float).T + b16.astype(float)   # the stored values
acc16 = np.mean(lr.classes_[scores.argmax(axis=1)] == y_te)
results.append(("logistic, 16 pixels, float16", kb((W16, b16, top)), acc16))
for name, size, acc in sorted(results, key=lambda r: -r[1]):
    print(f"{name:30s} {size:8.1f} KiB   accuracy {acc:.3f}")
