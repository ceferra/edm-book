import numpy as np
from ratings_toy import FILMS, R, USERS, pearson

# Item-based: precompute item-item similarities once (the columns of the matrix)
n = len(FILMS)
S = np.array([[pearson(R[:, i], R[:, j]) if i != j else -np.inf for j in range(n)]
              for i in range(n)])
print("films most similar to 'Space Saga':",
      ", ".join(f"{FILMS[j]} ({S[0, j]:+.2f})" for j in np.argsort(-S[0])[:2]))

def recommend_items(user):
    seen = ~np.isnan(R[user])
    scores = {}
    for i in np.flatnonzero(~seen):
        w = np.maximum(S[i, seen], 0)                  # similar items the user has rated
        if w.sum() > 0:
            scores[FILMS[i]] = float(w @ R[user, seen] / w.sum())
    return sorted(scores.items(), key=lambda kv: -kv[1])

print("item-based recommendations for Ivo:",
      ", ".join(f"{f} {s:.2f}" for f, s in recommend_items(USERS.index("Ivo"))))

# Recommending users to an item: who has not seen it, and would rate it highest?
film = FILMS.index("Silent Forest")
pred = {}
for u in np.flatnonzero(np.isnan(R[:, film])):
    w = np.array([max(pearson(R[u], R[v]), 0) if v != u and not np.isnan(R[v, film])
                  else 0.0 for v in range(len(USERS))])
    if w.sum() > 0:                                    # else: no evidence, no ranking
        pred[USERS[u]] = float(w @ np.nan_to_num(R[:, film]) / w.sum())
print("who should be told about 'Silent Forest':",
      ", ".join(f"{u} {s:.2f}" for u, s in sorted(pred.items(), key=lambda kv: -kv[1])))
