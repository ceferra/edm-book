import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from content_items import CATALOGUE

titles = list(CATALOGUE)
tfidf = TfidfVectorizer()
V = tfidf.fit_transform(CATALOGUE.values()).toarray()   # 10 films: dense is fine; keep
#                                                         # it sparse for large catalogues
V /= np.linalg.norm(V, axis=1, keepdims=True)
vocab = tfidf.vocabulary_
liked = {"Space Saga": 5, "Deep Ocean": 4, "Laugh Club": 1}   # one user's ratings
# likes push, dislikes pull
weights = np.array([liked[t] - 3 for t in liked])
profile = weights @ V[[titles.index(t) for t in liked]]         # the user's profile
if not profile.any():                       # likes and dislikes cancel out: no taste
    profile = V.mean(axis=0)                # to follow, so use a non-personalised ranking
terms = tfidf.get_feature_names_out()
print("strongest terms of the profile:", ", ".join(terms[np.argsort(-profile)[:5]]))
scores = {t: float(V[j] @ profile) for j, t in enumerate(titles) if t not in liked}
for t, s in sorted(scores.items(), key=lambda kv: -kv[1])[:2]:
    shared = [w for w in CATALOGUE[t].split() if w in vocab and profile[vocab[w]] > 0]
    print(f"  {t:12s} score {s:+.2f}   because of: {', '.join(shared)}")
print("Neither 'Last Signal' nor 'Red Planet' has a single rating: collaborative")
print("filtering cannot recommend them yet, a content-based recommender can.")
