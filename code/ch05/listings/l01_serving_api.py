import time
import numpy as np
from flask import Flask, jsonify, request
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

# Offline: train the model and fill a feature store with data known before any request
X, y = make_classification(n_samples=3000, n_features=4, n_informative=4, n_redundant=0,
                           random_state=0)
model = GradientBoostingClassifier(random_state=0).fit(X, y)
store = {f"c{i:04d}": X[i, 2:] for i in range(len(X))}       # history of each customer
VERSION = "1.3.0"
app = Flask(__name__)

@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):                          # 1. validate: structure,
        return jsonify(error="a JSON object is required"), 400
    if body.get("api_key") != "secret":                     #    permissions
        return jsonify(error="not authorised"), 401
    if not isinstance(body.get("customer"), str):
        return jsonify(error="customer must be a string"), 400
    try:                                                    #    and values
        if any(isinstance(body.get(k), bool) for k in ("f1", "f2")):
            raise TypeError
        x = np.array([float(body["f1"]), float(body["f2"])])
    except (KeyError, TypeError, ValueError, OverflowError):
        return jsonify(error="f1 and f2 must be numbers"), 400
    if not np.isfinite(x).all() or np.any(np.abs(x) > 10):
        return jsonify(error="f1 and f2 must be finite and in [-10, 10]"), 400
    history = store.get(body["customer"])                         # 2. gather more data
    if history is None:
        return jsonify(error="unknown customer"), 404
    features = np.r_[x, history].reshape(1, -1)                   # 3. preprocess
    p = float(model.predict_proba(features)[0, 1])                # 4. run the model
    return jsonify(probability=p, version=VERSION,                # 5. postprocess
                   decision="accept" if p >= 0.5 else "reject")   # 6. return

client = app.test_client()                 # in production: an HTTP server and clients
good = {"api_key": "secret", "customer": "c0007", "f1": 0.5, "f2": -1.2}
requests = [good, {**good, "f1": "high"}, {**good, "f1": "NaN"}, [1, 2],
            {**good, "customer": "c9999"}, {**good, "f2": 45},
            {**good, "api_key": "guess"}]
for body in requests:
    r = client.post("/predict", json=body)
    print(r.status_code, r.get_json())
start = time.perf_counter()
for _ in range(200):
    client.post("/predict", json=good)
print(f"mean latency: {(time.perf_counter() - start) / 200 * 1000:.1f} ms per request")
