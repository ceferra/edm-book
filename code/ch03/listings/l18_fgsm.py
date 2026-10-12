import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = load_digits(return_X_y=True)
X = X / 16.0                                                    # pixels in [0, 1]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

def fgsm(model, X, y, eps):
    """Fast gradient sign method for a softmax (multinomial logistic) model."""
    P = model.predict_proba(X)
    P[np.arange(len(y)), y] -= 1.0                              # d loss / d logits
    grad = P @ model.coef_                                       # d loss / d input
    return np.clip(X + eps * np.sign(grad), 0, 1)

lr = LogisticRegression(max_iter=3000).fit(X_tr, y_tr)
mlp = MLPClassifier((100,), max_iter=1000, random_state=0).fit(X_tr, y_tr)
rf = RandomForestClassifier(300, random_state=0).fit(X_tr, y_tr)
robust = LogisticRegression(max_iter=3000).fit(np.r_[X_tr, fgsm(lr, X_tr, y_tr, 0.15)],
                                               np.r_[y_tr, y_tr])    # adversarial training
print("eps   LR (white  MLP (trans-  RF (trans-  robust LR:   robust LR:")
print("      box)       ferred)      ferred)     old attack   adaptive")
for eps in (0.0, 0.05, 0.1, 0.2):
    Xa = fgsm(lr, X_te, y_te, eps)                   # crafted on the original LR
    Xr = fgsm(robust, X_te, y_te, eps)               # crafted on the robust LR itself
    print(f"{eps:.2f}  {lr.score(Xa, y_te):.3f}      {mlp.score(Xa, y_te):.3f}        "
          f"{rf.score(Xa, y_te):.3f}       {robust.score(Xa, y_te):.3f}        "
          f"{robust.score(Xr, y_te):.3f}")
