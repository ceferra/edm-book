import contextlib
import io
import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from jpmml_evaluator import make_evaluator
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

rng = np.random.default_rng(0)
n = 2000
X = pd.DataFrame({"income": rng.normal(30, 8, n).round(1),
                  "debt_ratio": rng.uniform(0, 0.8, n).round(2),
                  "years_job": rng.integers(0, 30, n)})
logit = -2 + 0.08 * X.income - 3 * X.debt_ratio + 0.03 * X.years_job
y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="repays")
tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=50, random_state=0)
pipe = PMMLPipeline([("tree", tree)]).fit(X, y)
pipe.verify(X.sample(10, random_state=0))       # embed records the engine must reproduce
path = os.path.join(tempfile.mkdtemp(), "loan_tree.pmml")
with contextlib.redirect_stdout(io.StringIO()):            # silence the converter's log
    sklearn2pmml(pipe, path)

# Inside the document: a data dictionary, and the tree as nested nodes with predicates
root = ET.parse(path).getroot()
ns = {"p": root.tag.split("}")[0][1:]}
fields = root.findall("p:DataDictionary/p:DataField", ns)
print(f"PMML {root.get('version')}, {os.path.getsize(path):,} bytes")
print("fields:", ", ".join(f"{f.get('name')} ({f.get('optype')})" for f in fields))

def show(node, depth=0):
    pred = node.find("p:SimplePredicate", ns)
    rule = "root" if depth == 0 else "otherwise" if pred is None else \
        f"{pred.get('field')} {pred.get('operator')} {float(pred.get('value')):.3g}"
    leaf = "" if node.find("p:Node", ns) is not None else f" -> {node.get('score')}"
    print(f"  {'  ' * depth}{rule}{leaf}")
    for child in node.findall("p:Node", ns):
        show(child, depth + 1)

show(root.find("p:TreeModel/p:Node", ns))

# Score the file with an independent engine (Java), which first checks the embedded
# verification records, and compare all the predictions with scikit-learn
print("elements:", [e.tag.split("}")[1] for e in root.find("p:TreeModel", ns)])
scores = make_evaluator(path).verify().evaluateAll(X)
same = np.array_equal(scores["repays"].to_numpy(), pipe.predict(X))
gap = np.abs(scores["probability(1)"].to_numpy() - pipe.predict_proba(X)[:, 1]).max()
print(f"JPMML on {n} applicants: same decisions {same}, largest difference {gap:.1e}")
