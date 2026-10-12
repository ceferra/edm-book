import numpy as np

# Probabilities of the classes (A, B) given by three models to one example
P = np.array([[0.90, 0.10],
              [0.45, 0.55],
              [0.40, 0.60]])
classes = np.array(["A", "B"])

votes = np.bincount(P.argmax(axis=1), minlength=2)   # majority voting
mean = P.mean(axis=0)                                 # linear combiner (uniform)
prod = P.prod(axis=0) / P.prod(axis=0).sum()          # product combiner (normalised)
print("votes:  ", votes, "->", classes[votes.argmax()])
print("mean:   ", mean.round(3), "->", classes[mean.argmax()])
print("product:", prod.round(3), "->", classes[prod.argmax()])
