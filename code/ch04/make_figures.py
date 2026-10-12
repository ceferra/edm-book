"""Generate all the figures of Unit 4 (Learning to learn).

Usage:
    python make_figures.py                     # writes ../../figures/ch04/*.pdf
    python make_figures.py OUT_DIR [fig_name ...]
All randomness is seeded. Several figures import the listings, so the numbers in the figures
are exactly those printed in the book.
"""
import os
import subprocess
import sys
import warnings
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ch01"))
sys.path.insert(0, os.path.join(HERE, "listings"))
from edm_style import (use_style, NAVY, BRONZE, TEAL, CORAL, GOLD, INDIGO, GRAY,  # noqa: E402
                       LIGHT, FULL, HALF)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "figures", "ch04")
warnings.filterwarnings("ignore")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"[saved] {name}.pdf")


def dot_to_pdf(src, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".dot")
    with open(path, "w") as fh:
        fh.write(src)
    subprocess.run(["dot", "-Tpdf", path, "-o", os.path.join(OUT, name + ".pdf")], check=True)
    os.remove(path)
    print(f"[saved] {name}.pdf (graphviz)")


# --------------------------------------------------------------------------- AutoML
def fig_hpo():
    X, y = make_classification(n_samples=600, n_features=10, n_informative=4, n_redundant=2,
                               class_sep=0.8, flip_y=0.05, random_state=11)
    gc, gg = np.linspace(-2, 3, 26), np.linspace(-4, 1, 26)
    cells = np.array([(c, g) for g in gg for c in gc])
    vals = np.array(Parallel(n_jobs=-1)(
        delayed(lambda z: cross_val_score(SVC(C=10 ** z[0], gamma=10 ** z[1]), X, y, cv=3).mean())(z)
        for z in cells))
    L = vals.reshape(len(gg), len(gc))
    B, rand_curves, bo_curves = 40, [], []
    for s in range(20):
        r = np.random.default_rng(s)
        order = r.permutation(len(vals))[:B]
        rand_curves.append(np.maximum.accumulate(vals[order]))
        ev = list(r.choice(len(vals), 5, replace=False))
        while len(ev) < B:
            gp = GaussianProcessRegressor(Matern(nu=2.5), normalize_y=True, alpha=1e-4)
            gp.fit(cells[ev], vals[ev])
            rest = np.setdiff1d(np.arange(len(vals)), ev)
            mu, sd = gp.predict(cells[rest], return_std=True)
            imp, sd = mu - vals[ev].max(), np.maximum(sd, 1e-9)
            ei = imp * norm.cdf(imp / sd) + sd * norm.pdf(imp / sd)
            ev.append(int(rest[np.argmax(ei)]))
        bo_curves.append(np.maximum.accumulate(vals[ev]))
        if s == 0:
            ev0, order0 = ev, order
    rand_curves, bo_curves = np.array(rand_curves), np.array(bo_curves)
    grid_pts = []
    for k in (2, 3, 4, 5, 6):
        idx = np.round(np.linspace(0, 25, k)).astype(int)
        grid_pts.append((k * k, L[np.ix_(idx, idx)].max()))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.75), gridspec_kw=dict(width_ratios=[1.05, 1]))
    fig.subplots_adjust(wspace=0.5)
    ax = axes[0]
    im = ax.imshow(L, origin="lower", extent=[-2.1, 3.1, -4.1, 1.1], aspect="auto", cmap="viridis",
                   vmin=0.5, vmax=vals.max())
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("CV accuracy", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.scatter(*cells[order0[:25]].T, marker="x", s=14, color="white", lw=0.9, label="random search")
    ax.scatter(*cells[ev0[:25]].T, s=12, color=CORAL, edgecolor="black", lw=0.4,
               label="Bayesian optimisation")
    best = np.argmax(vals)
    ax.plot(*cells[best], "*", ms=10, color=GOLD, mec="black", mew=0.5)
    ax.set_xlabel("$\\log_{10} C$")
    ax.set_ylabel("$\\log_{10}\\gamma$")
    ax.set_title("(a) Accuracy landscape, first 25 evaluations", fontsize=7.8)
    ax = axes[1]
    xs = np.arange(1, B + 1)
    for curves, c, lab in [(rand_curves, NAVY, "random search"), (bo_curves, CORAL, "Bayesian optimisation")]:
        ax.plot(xs, curves.mean(axis=0), color=c, lw=1.5, label=lab)
        ax.fill_between(xs, np.quantile(curves, 0.25, axis=0), np.quantile(curves, 0.75, axis=0),
                        color=c, alpha=0.15, lw=0)
    gx, gy = zip(*grid_pts)
    ax.plot(gx, gy, "s--", color=TEAL, ms=4, lw=1, label="grid search")
    ax.axhline(vals.max(), color=GRAY, ls=":", lw=1)
    ax.set_xlabel("number of evaluations")
    ax.set_ylabel("best accuracy found")
    ax.set_title("(b) Best so far (mean of 20 runs)", fontsize=7.8)
    ax.set_ylim(0.78, vals.max() + 0.006)
    ax.legend(loc="lower right", fontsize=6.3)
    for b in (10, 20, 40):
        print(f"  after {b} evaluations: random {rand_curves[:, b - 1].mean():.4f}, "
              f"BO {bo_curves[:, b - 1].mean():.4f}")
    print(f"  grid {grid_pts}; landscape max {vals.max():.4f}, "
          f"median {np.median(vals):.4f}, fraction >= max-0.01: {np.mean(vals >= vals.max() - 0.01):.3f}")
    save(fig, "hpo_search")


def fig_optimism():
    import l03_cash_optimism as m                   # evaluates the 200 configurations
    r = np.random.default_rng(1)
    Bs = np.unique(np.round(np.logspace(0, np.log10(200), 30)).astype(int))
    best_cv, sel_test = [], []
    for B in Bs:
        idx = [p[np.argmax(m.cv[p])] for p in (r.choice(200, B, replace=False) for _ in range(500))]
        best_cv.append(m.cv[idx].mean())
        sel_test.append(m.test[idx].mean())
    fig, ax = plt.subplots(figsize=(HALF + 0.7, 2.45))
    ax.plot(Bs, best_cv, "o-", ms=2.5, color=CORAL, label="cross-validation accuracy of the winner")
    ax.plot(Bs, sel_test, "s-", ms=2.5, color=NAVY, label="test accuracy of the winner")
    ax.axhline(m.test.max(), color=GRAY, ls=":", lw=1, label="best test accuracy of any configuration")
    ax.set_xscale("log")
    ax.set_xlabel("configurations evaluated by random search")
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower right", fontsize=6.2)
    save(fig, "automl_optimism")


def fig_metalearning():
    import l04_metalearning as m                    # 120 tasks, meta-features and the meta-model
    t = m.meta.tree_
    x_thr = t.threshold[0]
    right = t.children_right[0]
    d_thr = t.threshold[right]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.55), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(wspace=0.3)
    ax = axes[0]
    lin, d = m.M[:, 4], 10 ** m.M[:, 1]
    for w, c, lab in [(1, CORAL, "forest is better"), (0, TEAL, "logistic is better")]:
        sel = m.winner == w
        ax.scatter(lin[sel], d[sel], s=14, color=c, edgecolor="white", lw=0.3, label=lab)
    ax.axvline(x_thr, color=NAVY, lw=1)
    ax.plot([x_thr, 1.01], [10 ** d_thr] * 2, color=NAVY, lw=1)
    ax.text(x_thr + 0.005, 3, "choose the\nlogistic", fontsize=6.3, color=NAVY)
    ax.set_yscale("log")
    ax.set_xlabel("landmarker: accuracy of a linear discriminant")
    ax.set_ylabel("number of features")
    ax.set_title("(a) Tasks and the learned selection rule", fontsize=7.8)
    ax.legend(loc="lower left", fontsize=6.2, framealpha=0.9)
    ax = axes[1]
    names = ["always\nlogistic", "always\nforest", "meta-\nlearned", "oracle"]
    vals = [m.acc[:, 0].mean(), m.acc[:, 1].mean(), m.chosen.mean(), m.acc.max(axis=1).mean()]
    ax.bar(range(4), vals, color=[TEAL, CORAL, GOLD, GRAY], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylim(0.80, 0.872)
    ax.set_ylabel("mean accuracy over 120 tasks")
    ax.set_title("(b) Value of the selection", fontsize=7.8)
    print(f"  thresholds: lm_linear {x_thr:.3f}, log_d {d_thr:.3f} (d = {10 ** d_thr:.1f})")
    save(fig, "metalearning")


# --------------------------------------------------------------------------- genetic programming
def fig_gp():
    import l08_gp_symbolic as m
    h = np.array(m.history)
    gens = np.arange(len(h))
    fig, ax = plt.subplots(figsize=(HALF + 0.7, 2.35))
    ax.semilogy(gens, np.maximum(h[:, 1], 1e-3), "s-", ms=2.5, color=GRAY, label="median program")
    ax.semilogy(gens, np.maximum(h[:, 0], 1e-3), "o-", ms=2.5, color=CORAL, label="best program")
    ax.plot(gens[-1], 1e-3, "*", ms=11, color=GOLD, mec="black", mew=0.5)
    ax.annotate("exact: $x^2+xy-3$", (gens[-1], 1e-3), xytext=(gens[-1] - 7.5, 0.02), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("generation")
    ax.set_ylabel("mean squared error")
    ax.legend(loc="upper right", fontsize=6.5)
    save(fig, "gp_evolution")


# --------------------------------------------------------------------------- process mining
def fig_process():
    from event_log import ACTIVITIES, simulate_log
    from pm import alpha_miner, directly_follows
    log = simulate_log(500, seed=0).sort_values(["case", "timestamp"])
    code = {name: k for k, name in ACTIVITIES.items()}
    traces = list(log.groupby("case")["activity"].agg(lambda s: "".join(code[a] for a in s)))
    dfg = directly_follows(traces)
    counts = Counter(a for t in traces for a in t)
    log["next"] = log.groupby("case")["activity"].shift(-1)
    log["hours"] = (log.groupby("case")["timestamp"].shift(-1) - log["timestamp"]).dt.total_seconds() / 3600
    perf = log.dropna(subset=["next"]).groupby(["activity", "next"])["hours"].mean()
    starts, ends = Counter(t[0] for t in traces), Counter(t[-1] for t in traces)
    top = max(dfg.values())
    g = ['digraph G {', 'rankdir=LR; nodesep=0.22; ranksep=0.42; bgcolor="transparent";',
         'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12.5, '
         'color="#233A44", fillcolor="#E9EEF1", penwidth=0.8];',
         'edge [fontname="Helvetica", fontsize=10.5, arrowsize=0.6];',
         'start [shape=circle, label="", width=0.22, style=filled, fillcolor="#00796B", color="#00796B"];',
         'end [shape=doublecircle, label="", width=0.16, style=filled, fillcolor="#D9563F", color="#D9563F"];']
    for k, name in ACTIVITIES.items():
        g.append(f'{k} [label="{name}\\n{counts[k]}"];')
    for k, c in sorted(starts.items()):
        g.append(f'start -> {k} [label="{c}", color="#5A6B73", fontcolor="#5A6B73"];')
    for (a, b), c in sorted(dfg.items()):
        h = perf[(ACTIVITIES[a], ACTIVITIES[b])]
        col = "#D9563F" if h > 30 else "#5A6B73"
        g.append(f'{a} -> {b} [label="{c} | {h:.0f} h", penwidth={0.5 + 2.5 * c / top:.2f}, '
                 f'color="{col}", fontcolor="{col}"];')
    for k, c in sorted(ends.items()):
        g.append(f'{k} -> end [label="{c}", color="#5A6B73", fontcolor="#5A6B73"];')
    g.append("}")
    dot_to_pdf("\n".join(g), "process_dfg")
    places = alpha_miner(traces)
    p = ['digraph P {', 'rankdir=LR; nodesep=0.3; ranksep=0.32; bgcolor="transparent";',
         'node [fontname="Helvetica", fontsize=12.5];', 'edge [arrowsize=0.6, color="#233A44"];']
    for k, name in ACTIVITIES.items():
        p.append(f'{k} [shape=box, style="filled", fillcolor="#D7ECE9", color="#00796B", '
                 f'label="{name}", penwidth=0.8];')
    for i, (A, B) in enumerate(places):
        if not A:
            p.append(f'p{i} [shape=circle, label=<&#9679;>, fontsize=13, width=0.3, '
                     f'fixedsize=true, color="#233A44", xlabel="start"];')
            p.append(f'tok [shape=point, width=0.09, color="black"]; p{i} -> tok [style=invis];')
        elif not B:
            p.append(f'p{i} [shape=doublecircle, label="", width=0.2, color="#233A44", xlabel="end"];')
        else:
            p.append(f'p{i} [shape=circle, label="", width=0.28, fixedsize=true, color="#233A44"];')
        for a in sorted(A):
            p.append(f"{a} -> p{i};")
        for b in sorted(B):
            p.append(f"p{i} -> {b};")
    p.append("}")
    src = "\n".join(p).replace('p0 -> tok [style=invis];', '')
    src = src.replace('tok [shape=point, width=0.09, color="black"]; ', '')
    dot_to_pdf(src, "process_petri")
    print(f"  {len(dfg)} directly-follows edges; {len(places)} places")


ALL = [fig_hpo, fig_optimism, fig_metalearning, fig_gp, fig_process]

if __name__ == "__main__":
    use_style()
    only = set(sys.argv[2:])
    for f in ALL:
        if not only or f.__name__ in only:
            print(f.__name__)
            f()
