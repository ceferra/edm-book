import random
import numpy as np
import sympy

rng = random.Random(3)
x, y = np.random.default_rng(0).uniform(-3, 3, (2, 200))
target = x * x + x * y - 3                        # the hidden function to rediscover
OPS = {"+": np.add, "-": np.subtract, "*": np.multiply}
TERMINALS = ["x", "y", 1, 2, 3]

def random_tree(depth):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(TERMINALS)
    return (rng.choice(list(OPS)), random_tree(depth - 1), random_tree(depth - 1))

def evaluate(t):
    if isinstance(t, tuple):
        return OPS[t[0]](evaluate(t[1]), evaluate(t[2]))
    return x if t == "x" else y if t == "y" else np.full_like(x, t)

def error(t):
    return float(np.mean((evaluate(t) - target) ** 2))

def paths(t, path=()):                            # positions of all the subtrees
    yield path
    if isinstance(t, tuple):
        yield from paths(t[1], path + (1,))
        yield from paths(t[2], path + (2,))

def get(t, path):
    for i in path:
        t = t[i]
    return t

def put(t, path, new):
    if not path:
        return new
    return tuple(put(s, path[1:], new) if i == path[0] else s for i, s in enumerate(t))

def depth(t):
    return 1 + max(depth(t[1]), depth(t[2])) if isinstance(t, tuple) else 0

def crossover(a, b):                              # a subtree of b replaces one of a
    return put(a, rng.choice(list(paths(a))), get(b, rng.choice(list(paths(b)))))

def mutate(t):                                    # a random subtree becomes a new one
    return put(t, rng.choice(list(paths(t))), random_tree(2))

def tournament(errs, k=3):
    return min(rng.sample(range(len(errs)), k), key=lambda i: errs[i])

MAX_GEN = 60
pop, history = [random_tree(4) for _ in range(300)], []
for gen in range(MAX_GEN):
    errs = [error(t) for t in pop]
    best = min(range(len(pop)), key=errs.__getitem__)
    history.append((errs[best], float(np.median(errs))))
    if gen % 10 == 0 or errs[best] < 1e-12:
        print(f"generation {gen:2d}: best error {errs[best]:9.4f}, "
              f"median {history[-1][1]:9.1f}")
    if errs[best] < 1e-12 or gen == MAX_GEN - 1:
        break                          # pop[best] belongs to the evaluated population
    children = [pop[best]]                                        # elitism
    while len(children) < len(pop):
        child = crossover(pop[tournament(errs)], pop[tournament(errs)])
        if rng.random() < 0.2:
            child = mutate(child)
        if depth(child) <= 8:                                      # limit bloat
            children.append(child)
    pop = children
show = lambda t: f"({show(t[1])} {t[0]} {show(t[2])})" if isinstance(t, tuple) else str(t)
print("program:   ", show(pop[best]))
print("simplified:", sympy.expand(sympy.sympify(show(pop[best]))))
