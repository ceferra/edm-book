from scipy.stats import binom

def majority_accuracy(T, p):
    """P(more than half of T independent voters are right), for odd T."""
    return binom.sf(T // 2, T, p)

sizes = (1, 5, 11, 51, 101)
for p in (0.45, 0.6, 0.7):
    accs = [f"T={T}: {majority_accuracy(T, p):.3f}" for T in sizes]
    print(f"p={p:.2f}  " + "  ".join(accs))
