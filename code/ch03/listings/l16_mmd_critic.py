import numpy as np

def rbf(A, B, gamma):
    return np.exp(-gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))

def mmd_critic(X, m, c, gamma=0.5, radius=1.0):
    """MMD-critic (Kim et al., 2016): greedy prototypes and criticisms."""
    K = rbf(X, X, gamma)
    # mean similarity of each point to the data
    colmean = K.mean(axis=0)
    protos, path = [], []
    for _ in range(m):
        cands = [i for i in range(len(X)) if i not in protos]
        sets = [protos + [i] for i in cands]               # candidate prototype sets
        score = [K[np.ix_(s, s)].mean() - 2 * colmean[s].mean() for s in sets]
        protos.append(cands[int(np.argmin(score))])
        path.append(min(score) + K.mean())      # add the constant mean k(x,x')
    witness = colmean - K[:, protos].mean(axis=1)
    crit = []                          # most under-represented points, `radius` apart
    for i in np.argsort(-witness):
        if i not in protos and all(np.linalg.norm(X[i] - X[j]) > radius for j in crit):
            crit.append(i)
        if len(crit) == c:
            break
    return protos, crit, path, witness

rng = np.random.default_rng(5)
X = np.vstack([rng.normal([0, 0], 0.5, (240, 2)), rng.normal([3, 0.5], 0.6, (160, 2)),
               rng.normal([1.5, 3], 0.45, (90, 2)), rng.normal([4.2, 3.4], 0.25, (25, 2))])
protos, crit, path, witness = mmd_critic(X, m=4, c=3)
print("MMD^2 as prototypes are added:", np.round(path, 4))
print("prototypes:", np.round(X[protos], 1).tolist())
print("criticisms:", np.round(X[crit], 1).tolist())
print("their witness values:", np.round(witness[crit], 3))
