"""Synthetic ratings and three recommenders: biases, neighbours, matrix factorisation."""
import numpy as np

def make_ratings(n_users=600, n_items=400, k=5, seed=0, return_truth=False):
    """Ratings from latent tastes. Users rate popular items and items they like more
    often: the missing ratings are not missing at random."""
    rng = np.random.default_rng(seed)
    P, Q = rng.normal(0, 0.8, (n_users, k)), rng.normal(0, 0.8, (n_items, k))
    bu, bi = rng.normal(0, 0.4, n_users), rng.normal(0, 0.5, n_items)
    true = 3.3 + bu[:, None] + bi[None, :] + P @ Q.T
    pop = np.exp(1.2 * bi + rng.normal(0, 1.0, n_items))
    R_train = np.full((n_users, n_items), np.nan)
    test = []
    for u in range(n_users):
        w = pop * np.exp(0.5 * (P[u] @ Q.T))           # popularity and personal taste
        items = rng.choice(n_items, 8 + rng.poisson(15), replace=False, p=w / w.sum())
        r = np.clip(np.round(true[u, items] + rng.normal(0, 0.5, len(items))), 1, 5)
        order = rng.permutation(len(items))            # a random split of the history
        items, r = items[order], r[order]
        cut = max(1, len(items) // 5)                  # about 20% for testing
        R_train[u, items[cut:]] = r[cut:]
        test += [(u, i, x) for i, x in zip(items[:cut], r[:cut])]
    return (R_train, np.array(test), true) if return_truth else (R_train, np.array(test))

def fit_biases(R, reg=5.0, iterations=10):
    obs, Rz, mu = ~np.isnan(R), np.nan_to_num(R), np.nanmean(R)
    bu, bi = np.zeros(R.shape[0]), np.zeros(R.shape[1])
    for _ in range(iterations):                        # alternating regularised means
        bi = ((Rz - mu - bu[:, None]) * obs).sum(0) / (reg + obs.sum(0))
        bu = ((Rz - mu - bi[None, :]) * obs).sum(1) / (reg + obs.sum(1))
    return lambda u, i: np.clip(mu + bu[u] + bi[i], 1, 5)

def knn(R, by="item", k=30):
    """Neighbourhood model on mean-centred ratings (cosine of centred vectors)."""
    M = R.T if by == "item" else R                     # rows: the entities compared
    counts = np.sum(~np.isnan(M), axis=1)
    mu = np.where(counts > 0, np.nansum(M, axis=1) / np.maximum(counts, 1), np.nanmean(M))
    C = np.nan_to_num(M - mu[:, None])
    norm = np.linalg.norm(C, axis=1) + 1e-9
    S = C @ C.T / np.outer(norm, norm)
    np.fill_diagonal(S, 0)

    def predict(u, i):
        a, b = (i, u) if by == "item" else (u, i)      # predict entity a for column b
        cand = np.flatnonzero(~np.isnan(M[:, b]))
        top = cand[np.argsort(-S[a, cand])[:k]]
        w = S[a, top]
        if np.abs(w).sum() < 1e-9:
            return mu[a]
        return float(np.clip(mu[a] + w @ C[top, b] / np.abs(w).sum(), 1, 5))
    return predict

def fit_mf(R, k=10, lr=0.01, reg=0.05, epochs=40, seed=0):
    """Matrix factorisation with biases, trained by stochastic gradient descent."""
    rng = np.random.default_rng(seed)
    users, items = np.nonzero(~np.isnan(R))
    r, mu = R[users, items], np.nanmean(R)
    bu, bi = np.zeros(R.shape[0]), np.zeros(R.shape[1])
    P, Q = rng.normal(0, 0.1, (R.shape[0], k)), rng.normal(0, 0.1, (R.shape[1], k))
    for _ in range(epochs):
        for j in rng.permutation(len(r)):
            u, i = users[j], items[j]
            e = r[j] - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (e - reg * bu[u])
            bi[i] += lr * (e - reg * bi[i])
            pu = P[u].copy()
            P[u] += lr * (e * Q[i] - reg * P[u])
            Q[i] += lr * (e * pu - reg * Q[i])
    return lambda u, i: np.clip(mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=-1), 1, 5)
