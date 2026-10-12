"""The model behind the web application: training, saving and the answer shown."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PATH = Path(__file__).with_name("loan_model.joblib")

def train_and_save(path=PATH, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"income": rng.normal(30, 8, n), "debt": rng.uniform(0, 1, n)})
    p = 1 / (1 + np.exp(-(0.1 * (X.income - 30) - 2.5 * (X.debt - 0.3) - 0.5)))
    joblib.dump(LogisticRegression().fit(X, (rng.random(n) < p).astype(int)), path)

def answer_text(model, income, debt):
    """The sentence the application shows for an applicant."""
    x = pd.DataFrame({"income": [income], "debt": [debt]})
    return f"Probability of repaying: {model.predict_proba(x)[0, 1]:.2f}"
