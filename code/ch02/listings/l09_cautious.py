import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# (a) Rows = predicted a, b, c and "unknown"; columns = actual a, b, c
M = np.array([[19, 1, 2], [0, 30, 0], [0, 1, 38], [1, 2, 6]])
card, covered, hits = M.sum(), M[:3].sum(), np.trace(M[:3])
print(f"coverage={covered / card:.2f}  abstention={M[3].sum() / card:.2f}  "
      f"accuracy={hits / covered:.3f}  error={(covered - hits) / card:.2f}")

# (b) Does the confidence of each model rank its own mistakes?
X, y = make_classification(n_samples=4000, n_features=20, n_informative=6, n_redundant=4,
                           flip_y=0.05, class_sep=0.8, random_state=5)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
for name, m in [("logistic", LogisticRegression(max_iter=1000)),
                ("naive Bayes", GaussianNB()),
                ("tree (depth 6)", DecisionTreeClassifier(max_depth=6, random_state=0))]:
    P = m.fit(X_tr, y_tr).predict_proba(X_te)
    conf, correct = P.max(axis=1), P.argmax(axis=1) == y_te
    ok = correct[np.argsort(-conf, kind="stable")]            # most confident first
    risk = 1 - np.cumsum(ok) / np.arange(1, len(ok) + 1)      # error of the top-k
    acc20 = 1 - risk[int(0.8 * len(ok)) - 1]                  # with 20% abstention
    auc = roc_auc_score(correct, conf)                        # confidence vs correctness
    print(f"{name:14s} acc={correct.mean():.3f}  acc@20%abst={acc20:.3f}  "
          f"AURC={risk.mean():.3f}  AUC(conf)={auc:.3f}")
