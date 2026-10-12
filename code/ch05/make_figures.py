"""Generate all the figures of Unit 5 (Deployment of models).

Usage:
    python make_figures.py                     # writes ../../figures/ch05/*.pdf
    python make_figures.py OUT_DIR [fig_name ...]
The figures import the listings, so they show exactly the numbers printed in the book
(timings depend on the machine and change slightly from run to run).
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ch01"))
sys.path.insert(0, os.path.join(HERE, "listings"))
from edm_style import (use_style, NAVY, BRONZE, TEAL, CORAL, GOLD, INDIGO, GRAY,  # noqa: E402
                       LIGHT, FULL, HALF)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "figures", "ch05")
warnings.filterwarnings("ignore")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"[saved] {name}.pdf")


def fig_serving():
    import l02_latency as m
    lat = np.array(m.latency)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45))
    fig.subplots_adjust(wspace=0.32)
    ax = axes[0]
    ax.hist(lat, bins=40, range=(lat.min(), np.percentile(lat, 99.8)), color=TEAL, alpha=0.85)
    for q, c in [(50, NAVY), (95, GOLD), (99, CORAL)]:
        v = np.percentile(lat, q)
        ax.axvline(v, color=c, ls="--", lw=1, label=f"p{q} = {v:.1f} ms")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_xlabel("latency of one request (ms)")
    ax.set_ylabel("requests")
    ax.set_title("(a) Streaming: one prediction per call", fontsize=8)
    ax = axes[1]
    sizes = np.array(sorted(m.throughput))
    vals = np.array([m.throughput[s] for s in sizes])
    ax.loglog(sizes, vals, "o-", color=CORAL, ms=4)
    for s, v in zip(sizes, vals):
        ax.annotate(f"{v:,.0f}/s", (s, v), textcoords="offset points",
                    xytext=(6, 2) if s == 1 else (5, -9), fontsize=6.3)
    ax.set_xlim(0.6, 3e4)
    ax.set_xlabel("examples per call (batch size)")
    ax.set_ylabel("predictions per second")
    ax.set_title("(b) Batch: throughput grows with the batch", fontsize=8)
    save(fig, "serving_latency")


def fig_on_device():
    import l03_on_device as m
    colours = {"random forest": CORAL, "decision tree": GOLD, "neural network": TEAL,
               "logistic": NAVY}
    offsets = {"random forest, 300 trees": (-60, 6), "random forest, 30 trees": (-48, -12),
               "decision tree, depth 10": (6, -3), "neural network, float32": (6, -11),
               "neural network, float16": (-14, 8), "neural network, int8": (-58, -12),
               "logistic, 16 pixels, float16": (6, -3)}
    fig, ax = plt.subplots(figsize=(HALF + 1.3, 2.7))
    for name, kb, acc in m.results:
        c = next(v for k, v in colours.items() if name.startswith(k))
        ax.scatter(kb, acc, s=28, color=c, edgecolor="black", lw=0.4, zorder=3)
        ax.annotate(name, (kb, acc), textcoords="offset points", xytext=offsets[name], fontsize=6.2)
    pts = sorted((kb, acc) for _, kb, acc in m.results)
    front, best = [], -1
    for kb, acc in pts:
        if acc > best:
            front.append((kb, acc))
            best = acc
    front.append((4e4, best))
    fx, fy = zip(*front)
    ax.step(fx, fy, where="post", color=GRAY, ls=":", lw=1, label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlim(0.35, 4e4)
    ax.set_ylim(0.82, 0.99)
    ax.set_xlabel("size of the model shipped to the device (KiB, log scale)")
    ax.set_ylabel("test accuracy")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "on_device")


def fig_federated():
    import l04_federated as m
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw=dict(width_ratios=[1.05, 1]))
    fig.subplots_adjust(wspace=0.3)
    ax = axes[0]
    rounds = np.arange(1, len(m.history) + 1)
    ax.plot(rounds, m.history, "o-", ms=2.5, color=TEAL, label="federated averaging")
    ax.axhline(m.results["centralised (data pooled)"][0], color=NAVY, ls="--", lw=1,
               label="centralised (data pooled)")
    ax.axhline(m.results["local models only"][0], color=CORAL, ls=":", lw=1.2,
               label="local models only (mean)")
    ax.set_xlabel("communication round")
    ax.set_ylabel("accuracy on all digits")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("(a) A global model without sharing data", fontsize=8)
    ax.legend(loc="lower right", fontsize=6.3)
    ax = axes[1]
    names = list(m.results)
    short = ["central-\nised", "local\nonly", "feder-\nated", "federated\n+ tuning"]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [m.results[n][0] for n in names], 0.4, color=NAVY, label="all digits")
    ax.bar(x + 0.2, [m.results[n][1] for n in names], 0.4, color=GOLD, label="each user's own data")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6.6)
    ax.set_ylim(0.5, 1.07)
    ax.set_ylabel("accuracy")
    ax.set_title("(b) Global versus personalised performance", fontsize=8)
    ax.legend(loc="upper center", ncol=2, fontsize=6.2, framealpha=0.95)
    save(fig, "federated")


def fig_continuous():
    import l07_continuous_training as m
    h = m.history
    weeks = np.array([r[0] for r in h])
    fig, ax = plt.subplots(figsize=(HALF + 1.3, 2.55))
    ax.axvspan(5.6, 6.4, color=CORAL, alpha=0.12, lw=0)
    ax.text(6, 0.925, "labels lost\nin week 6", ha="center", fontsize=6.3, color=CORAL)
    for idx, c, mk, lab in [(1, GRAY, "s", "deployed once (static)"),
                            (2, GOLD, "^", "retrained without checks"),
                            (3, TEAL, "o", "continuous training with tests and gate")]:
        ax.plot(weeks, [r[idx] for r in h], marker=mk, ms=3.2, color=c, lw=1.3, label=lab)
    ax.set_xlabel("week")
    ax.set_ylabel("accuracy in production")
    ax.set_ylim(0.48, 0.95)
    ax.set_xticks(weeks)
    ax.legend(loc="lower left", fontsize=6.3)
    save(fig, "continuous_training")


ALL = [fig_serving, fig_on_device, fig_federated, fig_continuous]

if __name__ == "__main__":
    use_style()
    only = set(sys.argv[2:])
    for f in ALL:
        if not only or f.__name__ in only:
            print(f.__name__)
            f()
