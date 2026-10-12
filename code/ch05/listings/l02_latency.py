import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=20000, n_features=20, random_state=0)
model = RandomForestClassifier(200, n_jobs=1, random_state=0).fit(X[:5000], y[:5000])
pool = X[5000:]

latency = []                                       # streaming: one request per call
for i in range(500):
    start = time.perf_counter()
    model.predict_proba(pool[i:i + 1])
    latency.append((time.perf_counter() - start) * 1000)
p50, p95, p99 = np.percentile(latency, [50, 95, 99])
print(f"single requests: p50 {p50:.1f} ms, p95 {p95:.1f} ms, p99 {p99:.1f} ms")

throughput = {}                                    # batch: many examples per call
for size in (1, 10, 100, 1000, 10000):
    reps = max(3, 300 // size)
    start = time.perf_counter()
    for _ in range(reps):
        model.predict_proba(pool[:size])
    throughput[size] = size * reps / (time.perf_counter() - start)
    print(f"batches of {size:5d}: {throughput[size]:9,.0f} predictions per second")
