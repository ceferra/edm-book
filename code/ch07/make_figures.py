"""Generate all the figures of Unit 7 (Applications).

Usage:
    python make_figures.py                     # writes ../../figures/ch07/*.pdf
    python make_figures.py OUT_DIR [fig_name ...]
The figures import the listings, so they show the numbers printed in the book.
"""
import contextlib
import io
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

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "figures", "ch07")
warnings.filterwarnings("ignore")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"[saved] {name}.pdf")


def quiet_import(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return __import__(name)


def fig_similarity():
    from ratings_toy import FILMS, R, USERS
    me, ana = USERS.index("Ivo"), USERS.index("Ana")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    fig.subplots_adjust(wspace=0.32)
    ax = axes[0]
    both = ~np.isnan(R[me]) & ~np.isnan(R[ana])
    x, y = R[ana, both], R[me, both]
    ax.scatter(x, y, color=NAVY, s=22, zorder=3)
    for xi, yi, f in zip(x, y, np.array(FILMS)[both]):
        ax.annotate(f, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=6.5)
    a, b = np.polyfit(x, y, 1)
    xx = np.linspace(1, 5, 2)
    ax.plot(xx, a * xx + b, color=CORAL, lw=1.2)
    ax.set_xlim(1, 5.4)
    ax.set_ylim(1, 5.4)
    ax.set_xlabel("Ana's rating")
    ax.set_ylabel("Ivo's rating")
    ax.set_title(f"(a) Pearson correlation {np.corrcoef(x, y)[0, 1]:.2f}", fontsize=8)
    ax = axes[1]
    i, j = FILMS.index("Space Saga"), FILMS.index("Laugh Club")
    for u, name in enumerate(USERS):
        if np.isnan(R[u, i]) or np.isnan(R[u, j]):
            continue
        c = CORAL if name == "Ivo" else NAVY
        ax.scatter(R[u, i], R[u, j], color=c, s=22, zorder=3)
        ax.annotate(name, (R[u, i], R[u, j]), textcoords="offset points", xytext=(4, 4),
                    fontsize=6.5, color=c)
        if name != "Ivo":
            ax.plot([R[me, i], R[u, i]], [R[me, j], R[u, j]], color=GRAY, lw=0.6, ls=":")
    ax.set_xlim(1, 5.6)
    ax.set_ylim(1, 5.6)
    ax.set_xlabel("rating of Space Saga")
    ax.set_ylabel("rating of Laugh Club")
    ax.set_title("(b) Users as points: Euclidean distance", fontsize=8)
    save(fig, "similarity")


def fig_recsys_eval():
    l03 = quiet_import("l03_rating_prediction")
    l05 = quiet_import("l05_topn_evaluation")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5))
    fig.subplots_adjust(wspace=0.55)
    ax = axes[0]
    names = list(l03.results)
    vals = [l03.results[n] for n in names]
    ax.plot(vals, range(len(names)), "o", color=NAVY, ms=5)
    for k, v in enumerate(vals):
        ax.text(v + 0.004, k, f"{v:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(min(vals) - 0.02, max(vals) + 0.04)
    ax.set_xlabel("RMSE on held-out ratings (lower is better)")
    ax.set_title("(a) Predicting ratings", fontsize=8)
    ax = axes[1]
    colours = {"random": GRAY, "most popular": BRONZE, "matrix factorisation": NAVY,
               "item co-occurrence": TEAL}
    offsets = {"random": (5, 3), "most popular": (5, -9), "matrix factorisation": (-10, -11),
               "item co-occurrence": (5, 3)}
    for name, (p, r, n, c, nov, orc) in l05.metrics.items():
        ax.scatter(p, orc, s=30, color=colours[name], zorder=3)
        ax.annotate(name, (p, orc), textcoords="offset points", xytext=offsets[name], fontsize=6.5)
    ax.set_xlabel("precision@10 on observed test positives")
    ax.set_ylabel("expected precision@10 (simulator)")
    ax.set_xlim(-0.005, 0.11)
    ax.set_ylim(0.4, 0.75)
    ax.set_title("(b) What is measured matters", fontsize=8)
    save(fig, "recsys_eval")


def fig_campaign():
    m = quiet_import("l06_mailing_campaign")
    share = np.arange(1, len(m.ranked) + 1) / len(m.ranked)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5))
    fig.subplots_adjust(wspace=0.32)
    ax = axes[0]
    ax.plot(share, m.gains, color=NAVY, lw=1.4, label="model")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1, label="random")
    perfect = np.minimum(share / m.ranked.mean(), 1)
    ax.plot(share, perfect, ":", color=TEAL, lw=1, label="perfect")
    ax.set_xlabel("share of customers mailed")
    ax.set_ylabel("share of all responses")
    ax.set_title("(a) Cumulative response (gains)", fontsize=8)
    ax.legend(loc="lower right", fontsize=6.3)
    ax = axes[1]
    for (name, profit), c in zip(m.curves.items(), [CORAL, INDIGO]):
        x = np.arange(len(profit)) / (len(profit) - 1)       # 0: do not launch
        ax.plot(x, profit / 1000, color=c, lw=1.4, label=name)
        best = int(np.argmax(profit))
        ax.plot(x[best], profit[best] / 1000, "o", color=c, ms=4)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("share of customers mailed")
    ax.set_ylabel("estimated profit (thousand EUR)")
    ax.set_title("(b) Profit for two cost configurations", fontsize=8)
    ax.legend(loc="lower center", fontsize=6.3)
    save(fig, "campaign")


def fig_uplift():
    m = quiet_import("l07_uplift")
    fig, ax = plt.subplots(figsize=(HALF + 1.2, 2.5))
    k = np.arange(1, len(m.idx) + 1)
    for (name, s), c in zip(m.scores.items(), [GRAY, CORAL, TEAL]):
        order = m.idx[np.argsort(-s[m.idx])]
        ax.plot(k / len(k), np.cumsum(m.tau[order]), color=c, lw=1.4, label=name)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlabel("share of customers contacted")
    ax.set_ylabel("extra sales caused (cumulative)")
    ax.legend(loc="upper left", fontsize=6.3)
    save(fig, "uplift")


ALL = [fig_similarity, fig_recsys_eval, fig_campaign, fig_uplift]

if __name__ == "__main__":
    use_style()
    only = set(sys.argv[2:])
    for f in ALL:
        if not only or f.__name__ in only:
            print(f.__name__)
            f()
