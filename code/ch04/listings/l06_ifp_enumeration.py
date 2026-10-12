from functools import lru_cache
from itertools import product

LEAVES, FUNCTIONS = ("x", "1", "2"), ("+", "*")           # the background knowledge

@lru_cache(maxsize=None)
def programs(k):
    """All expressions with exactly k functions, as fully parenthesised strings."""
    if k == 0:
        return LEAVES
    return tuple(f"({a} {f} {b})" for left in range(k)
                 for a, b, f in product(programs(left), programs(k - 1 - left),
                                        FUNCTIONS))

examples = [(6, 14), (3, 8), (5, 12), (2, 6)]               # input -> output
print("programs with 0..4 functions:", [len(programs(k)) for k in range(5)])
for k in range(5):
    hits = [p for p in programs(k) if all(eval(p, {"x": x}) == y for x, y in examples)]
    if hits:
        break
evaluated = sum(len(programs(j)) for j in range(k + 1))
print(f"{len(hits)} consistent programs with {k} functions "
      f"({evaluated} programs evaluated)")
print("e.g.", ", ".join(hits[:3]))
print("for input 4 they output", eval(hits[0], {"x": 4}))
