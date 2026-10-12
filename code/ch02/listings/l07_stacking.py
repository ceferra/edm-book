from sklearn.datasets import make_classification
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=2000, n_features=20, n_informative=6, n_redundant=4,
                           flip_y=0.05, class_sep=0.8, random_state=11)
base = [("logistic", make_pipeline(StandardScaler(), LogisticRegression())),
        ("naive Bayes", GaussianNB()),
        ("15-NN", make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
        ("tree", DecisionTreeClassifier(max_depth=6, random_state=0))]
models = dict(base)
models["hard voting"] = VotingClassifier(base, voting="hard")
models["soft voting"] = VotingClassifier(base, voting="soft")
models["stacking"] = StackingClassifier(base, final_estimator=LogisticRegression(), cv=5)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, m in models.items():
    scoring = ["accuracy", "roc_auc"] if hasattr(m, "predict_proba") else ["accuracy"]
    r = cross_validate(m, X, y, cv=cv, scoring=scoring)
    auc = f"{r['test_roc_auc'].mean():.3f}" if "roc_auc" in scoring else "  -  "
    print(f"{name:12s} accuracy={r['test_accuracy'].mean():.3f}  AUC={auc}")
