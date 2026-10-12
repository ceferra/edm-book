from itertools import combinations, product

# Candidate body literals over the head variables X, Y and an extra variable Z
LITERALS = [("parent", v) for v in product("XYZ", repeat=2) if v[0] != v[1]] + \
           [(p, (v,)) for p in ("male", "female") for v in "XYZ"]

def covers(body, x, y, B, people):
    """Does the body entail the head p(x, y) for some value of Z?"""
    return any(all(tuple({"X": x, "Y": y, "Z": z}[v] for v in args) in B[pred]
                   for pred, args in body) for z in people)

def learn(positives, B):
    """Shortest body that covers all positives and no negative (closed world)."""
    people = sorted({c for facts in B.values() for f in facts for c in f})
    negatives = [(x, y) for x in people for y in people if (x, y) not in positives]
    for size in (1, 2):
        for body in combinations(LITERALS, size):
            if all(covers(body, *e, B, people) for e in positives) and \
               not any(covers(body, *e, B, people) for e in negatives):
                return body

show = lambda head, body: f"{head}(X,Y) :- " + ", ".join(
    f"{p}({','.join(a)})" for p, a in body)
B = {"parent": {("a", "b"), ("a", "c"), ("d", "b")},              # background knowledge
     "male": {("a",)}, "female": {("c",), ("d",)}}
print(show("father", learn({("a", "b"), ("a", "c")}, B)))
print(show("mother", learn({("d", "b")}, B)))
B2 = {"parent": {("ann", "bob"), ("bob", "carl"), ("bob", "dora"), ("eve", "fred"),
                 ("fred", "gina"), ("eve", "hugo")}, "male": set(), "female": set()}
print(show("grandparent", learn({("ann", "carl"), ("ann", "dora"), ("eve", "gina")}, B2)))
n_bodies = len(LITERALS) * (len(LITERALS) + 1) // 2
print(f"{len(LITERALS)} literals: {n_bodies} bodies of size <= 2")
