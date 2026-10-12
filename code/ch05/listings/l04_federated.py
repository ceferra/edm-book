import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X = X / 16.0
X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=0,
                                                  stratify=y)
K = 10                                             # clients, e.g. phones of 10 users
favourite = (y_pool + rng.integers(0, 2, len(y_pool))) % K   # each digit: 2 clients
to_favourite = rng.random(len(y_pool)) < 0.8                 # 80% go to a favourite
owner = np.where(to_favourite, favourite, rng.integers(0, K, len(y_pool)))
clients = []                                       # non-IID data: (train, own test)
for k in range(K):
    idx = rng.permutation(np.flatnonzero(owner == k))
    cut = int(0.7 * len(idx))
    clients.append((X_pool[idx[:cut]], y_pool[idx[:cut]],
                    X_pool[idx[cut:]], y_pool[idx[cut:]]))

def sgd(W, X, y, epochs, lr=0.5, batch=16):
    """Mini-batch SGD for softmax regression; the last row of W is the bias."""
    W, Xb, Y = W.copy(), np.c_[X, np.ones(len(X))], np.eye(10)[y]
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            j = perm[s:s + batch]
            Z = Xb[j] @ W
            P = np.exp(Z - Z.max(axis=1, keepdims=True))
            W -= lr * Xb[j].T @ (P / P.sum(axis=1, keepdims=True) - Y[j]) / len(j)
    return W

acc = lambda W, X, y: np.mean(np.argmax(np.c_[X, np.ones(len(X))] @ W, axis=1) == y)
own = lambda models: np.mean([acc(W, c[2], c[3]) for W, c in zip(models, clients)])
W0, n = np.zeros((65, 10)), np.array([len(c[1]) for c in clients])
W, history = W0, []
for rnd in range(30):                              # federated averaging (FedAvg)
    local = [sgd(W, c[0], c[1], epochs=2) for c in clients]      # on the devices
    W = sum(nk * Wk for nk, Wk in zip(n, local)) / n.sum()       # on the server
    history.append(acc(W, X_test, y_test))
central = sgd(W0, np.vstack([c[0] for c in clients]),
              np.concatenate([c[1] for c in clients]), epochs=60)
alone = [sgd(W0, c[0], c[1], epochs=60) for c in clients]
tuned = [sgd(W, c[0], c[1], epochs=2) for c in clients]          # personalisation
print("model                        all digits   each user's own data")
results = {}
for name, models in [("centralised (data pooled)", [central] * K),
                     ("local models only", alone), ("federated averaging", [W] * K),
                     ("federated + fine-tuning", tuned)]:
    everyone = np.mean([acc(m, X_test, y_test) for m in models])
    results[name] = (everyone, own(models))
    print(f"{name:28s}    {everyone:.3f}          {own(models):.3f}")
print(f"each round: {K} clients x {W.size} parameters each way; no data leaves a device")
