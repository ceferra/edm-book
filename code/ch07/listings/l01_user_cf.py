import numpy as np
from ratings_toy import FILMS, R, USERS, euclidean_similarity, pearson

me = USERS.index("Ivo")
print("similarity of Ivo to   " + "  ".join(f"{u:>6s}" for u in USERS[:-1]))
for name, sim in [("Euclidean-based", euclidean_similarity), ("Pearson", pearson)]:
    print(f"{name:22s}" + "  ".join(f"{sim(R[me], R[v]):6.2f}" for v in range(me)))

def recommend(user, sim=pearson):
    """Weighted average of the others' ratings, with positive similarities as weights."""
    rows = []
    for i, film in enumerate(FILMS):
        if not np.isnan(R[user, i]):
            continue                                   # already seen
        total = sim_sum = 0.0
        for v in range(len(USERS)):
            s = sim(R[user], R[v]) if v != user else 0.0
            if s > 0 and not np.isnan(R[v, i]):
                total += s * R[v, i]
                sim_sum += s
        rows.append((film, total, sim_sum, total / sim_sum if sim_sum else np.nan))
    return sorted(rows, key=lambda r: -r[3])

print("film            total   sim. sum   predicted rating")
for film, total, sim_sum, score in recommend(me):
    print(f"{film:14s} {total:6.2f}   {sim_sum:6.2f}       {score:.2f}")
