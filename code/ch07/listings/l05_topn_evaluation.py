import numpy as np
from scipy.stats import norm as gaussian
from recsys_models import fit_mf, make_ratings

R, test, true = make_ratings(return_truth=True)   # true: the simulator's latent ratings
n_users, n_items = R.shape
seen = ~np.isnan(R)
relevant = [set() for _ in range(n_users)]
for u, i, r in test:
    if r >= 4:                                         # held-out items the user liked
        relevant[int(u)].add(int(i))
popularity = seen.sum(axis=0)

# A score for every user and item from four recommenders
B = seen.astype(float)                                 # implicit feedback: who rated what
cooc = B.T @ B                                         # item co-occurrence
norm = np.sqrt(np.diag(cooc)) + 1e-9
S = cooc / np.outer(norm, norm)                        # cosine similarity between items
np.fill_diagonal(S, 0)
mf = fit_mf(R)
scores = {"random": np.random.default_rng(1).random((n_users, n_items)),
          "most popular": np.tile(popularity.astype(float), (n_users, 1)),
          "matrix factorisation": mf(np.arange(n_users)[:, None],
                                     np.arange(n_items)[None, :]),
          "item co-occurrence": B @ S}

def evaluate(score, k=10):
    k = min(k, int((~seen).sum(axis=1).min()))     # never recommend an item already seen
    top = np.argsort(-np.where(seen, -np.inf, score), axis=1)[:, :k]   # unseen items only
    prec, rec, ndcg = [], [], []
    for u in range(n_users):
        if not relevant[u]:
            continue
        hits = np.array([i in relevant[u] for i in top[u]], dtype=float)
        prec.append(hits.mean())                      # macro-average over users
        rec.append(hits.sum() / len(relevant[u]))
        ideal = np.sum(1 / np.log2(np.arange(2, min(k, len(relevant[u])) + 2)))
        ndcg.append(np.sum(hits / np.log2(np.arange(2, k + 2))) / ideal)
    # share of the catalogue recommended
    coverage = len(np.unique(top)) / n_items           # share of the catalogue
    share = (popularity[top] + 1) / (popularity.sum() + n_items)   # smoothed popularity
    if not prec:                                      # nobody has positive test items
        return (np.nan,) * 6
    users = [u for u in range(n_users) if relevant[u]]
    # oracle: probability that the user would rate each recommended item 4 or more
    latent = np.take_along_axis(true[users], top[users], axis=1)
    liked = gaussian.cdf((latent - 3.5) / 0.5)
    return (np.mean(prec), np.mean(rec), np.mean(ndcg), coverage,
            np.mean(-np.log2(share)), liked.mean())

metrics = {name: evaluate(score) for name, score in scores.items()}
print(f"{sum(map(bool, relevant))} of {n_users} users have liked test items")
print("(P, R, NDCG: observed positives; oracle: the simulator's latent ratings)")
print("recommender           P@10   R@10   NDCG@10  coverage  novelty  oracle P@10")
for name, (p, r, n, c, nov, orc) in metrics.items():
    print(f"{name:21s} {p:.3f}  {r:.3f}  {n:.3f}    {c:6.1%}    {nov:5.1f}"
          f"      {orc:.3f}")
