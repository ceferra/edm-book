import numpy as np

rng = np.random.default_rng(3)
n_items, runs = 400, 5
difficulty = rng.normal(0, 1.5, n_items)           # some questions are much harder
systems = {"A (small model)": (0.8, 0.002),         # ability, cost per question in EUR
           "B (large model)": (1.1, 0.010)}
results = {}
for name, (ability, cost) in systems.items():
    p = 1 / (1 + np.exp(-(ability - difficulty)))   # chance of a right answer (IRT)
    results[name] = rng.random((runs, n_items)) < p  # five runs with sampling: answers vary
    acc = results[name].mean(axis=1)
    print(f"{name}: accuracy in each run {np.round(acc, 3)}")
    print(f"  mean {acc.mean():.3f}, cost per right answer {cost / acc.mean():.4f} EUR")

a, b = (r.mean(axis=0) for r in results.values())   # mean score of each question over runs
diff = b - a
se_unpaired = np.sqrt(a.var(ddof=1) / n_items + b.var(ddof=1) / n_items)
se_paired = diff.std(ddof=1) / np.sqrt(n_items)
print(f"B - A = {diff.mean():+.3f}; 95% interval unpaired +/-{1.96 * se_unpaired:.3f}, "
      f"paired by question +/-{1.96 * se_paired:.3f}")
single = [results["B (large model)"][i].mean() - results["A (small model)"][j].mean()
          for i in range(runs) for j in range(runs)]
print(f"single-run comparisons: B - A between {min(single):+.3f} and {max(single):+.3f}")
