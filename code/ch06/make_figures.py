"""Generate all the figures of Unit 6 (Monitoring and updating models).

Usage:
    python make_figures.py                     # writes ../../figures/ch06/*.pdf
    python make_figures.py OUT_DIR [fig_name ...]
Several figures import the listings, so they show the numbers printed in the book.
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
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ch01"))
sys.path.insert(0, os.path.join(HERE, "listings"))
from edm_style import (use_style, NAVY, BRONZE, TEAL, CORAL, GOLD, INDIGO, GRAY,  # noqa: E402
                       LIGHT, FULL, HALF)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "figures", "ch06")
warnings.filterwarnings("ignore")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"[saved] {name}.pdf")


def quiet_import(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return __import__(name)


def fig_shift_types():
    rng = np.random.default_rng(0)
    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    def source(n, prior=0.5):
        y = (rng.random(n) < prior).astype(int)
        return rng.normal(size=(n, 2)) + np.c_[2 * y - 1, np.zeros(n)], y

    X_s, y_s = source(4000)
    model = LogisticRegression().fit(X_s, y_s)
    Xc = rng.normal([1.2, 1.5], [0.6, 0.8], size=(4000, 2))           # new p(x)
    yc = (rng.random(4000) < sigmoid(2 * Xc[:, 0])).astype(int)       # same p(y|x)
    Xp, yp = source(4000, prior=0.1)                                  # new p(y)
    Xd = rng.normal(size=(4000, 2)) * 1.3                             # new p(y|x)
    yd = (rng.random(4000) < sigmoid(2 * Xd[:, 1])).astype(int)
    panels = [("(a) training data", X_s, y_s), ("(b) covariate shift", Xc, yc),
              ("(c) prior (label) shift", Xp, yp), ("(d) concept shift", Xd, yd)]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.05), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08)
    xx = np.linspace(-4, 4, 2)
    w, b = model.coef_[0], model.intercept_[0]
    for ax, (title, X, y) in zip(axes, panels):
        idx = rng.choice(len(y), 500, replace=False)
        ax.scatter(*X[idx][y[idx] == 0].T, s=3, color=TEAL, alpha=0.6)
        ax.scatter(*X[idx][y[idx] == 1].T, s=3, color=CORAL, alpha=0.6)
        ax.plot(xx, -(b + w[0] * xx) / w[1], color=NAVY, lw=1.3)
        acc = model.score(X, y)
        extra = ""
        if title.startswith("(c)"):
            adjusted = np.mean(((X @ w + b + np.log(0.1 / 0.9)) > 0) == y)
            extra = f" ({adjusted:.2f} adjusted)"
            print(f"  accuracy with the threshold adjusted to the new prior {adjusted:.3f}")
        ax.set_title(f"{title}\naccuracy {acc:.2f}{extra}", fontsize=7.2)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  {title}: accuracy of the training model {acc:.3f}")
    shift = np.log(0.1 / 0.9)                                          # Bayes boundary, new prior
    axes[2].plot(xx, -(b + shift + w[0] * xx) / w[1], "--", color=GOLD, lw=1.2)
    axes[3].axhline(0, color=GOLD, ls="--", lw=1.2)
    save(fig, "shift_types")


def fig_drift_patterns():
    t = np.linspace(0, 1, 400)
    rng = np.random.default_rng(1)
    gradual = (rng.random(400) < np.clip((t - 0.3) / 0.4, 0, 1)).astype(float)
    curves = [("sudden", (t > 0.5).astype(float)), ("gradual", gradual),
              ("incremental", np.clip((t - 0.3) / 0.4, 0, 1)),
              ("recurring", ((t > 0.25) & (t < 0.5) | (t > 0.75)).astype(float))]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.45), sharey=True)
    fig.subplots_adjust(wspace=0.1, bottom=0.22)
    for ax, (name, c) in zip(axes, curves):
        if name == "gradual":
            ax.scatter(t, c, s=2, color=CORAL)
        else:
            ax.plot(t, c, color=CORAL, lw=1.5)
        ax.set_title(name, fontsize=7.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["old", "new"], fontsize=6.5)
        ax.set_xticks([])
        ax.set_xlabel("time", fontsize=7)
        ax.set_ylim(-0.25, 1.25)
    save(fig, "drift_patterns")


def fig_concept_drift():
    m = quiet_import("l03_concept_drift")
    fig, ax = plt.subplots(figsize=(FULL, 2.4))
    ax.axvline(2000, color=GRAY, ls=":", lw=1)
    ax.axvspan(4000, 5000, color=GOLD, alpha=0.12, lw=0)
    ax.text(2330, 0.70, "sudden drift", fontsize=6.5, color=GRAY)
    ax.text(4030, 0.70, "incremental drift", fontsize=6.5, color=BRONZE)
    styles = {"static model": (GRAY, "-"), "sliding window": (TEAL, "-"),
              "DDM": (CORAL, "-"), "Page-Hinkley": (INDIGO, "--")}
    k = 200
    for name, (errors, alarms) in m.results.items():
        acc = 1 - np.convolve(errors, np.ones(k) / k, mode="valid")
        c, ls = styles[name]
        ax.plot(np.arange(500 + k - 1, m.T), acc, color=c, ls=ls, lw=1.1, label=name)
        for a in alarms:
            ax.plot(a, 0.98, "v", color=c, ms=4)
    ax.set_xlim(500, m.T)
    ax.set_ylim(0.15, 1.02)
    ax.set_xlabel("example (stream position)")
    ax.set_ylabel(f"accuracy (last {k})")
    ax.legend(loc="lower left", fontsize=6.3, ncol=4)
    save(fig, "concept_drift")


def fig_covariate_shift():
    m = quiet_import("l05_covariate_shift")
    from scipy.stats import norm
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45))
    fig.subplots_adjust(wspace=0.28)
    ax = axes[0]
    ax.scatter(m.x_s, m.y_s, s=6, color=NAVY, alpha=0.6, label="training data")
    sub = np.random.default_rng(0).choice(len(m.x_t), 200, replace=False)
    ax.scatter(m.x_t[sub], m.y_t[sub], s=6, color=GOLD, alpha=0.7, label="deployment data")
    xx = np.linspace(-1.2, 2.0, 200)
    ax.plot(xx, m.f(xx), color=GRAY, lw=1, ls=":", label="true function")
    ax.plot(xx, m.fits["unweighted"].predict(xx[:, None]), color=CORAL, lw=1.3, label="unweighted fit")
    ax.plot(xx, m.fits["weighted (estimated w)"].predict(xx[:, None]), color=TEAL, lw=1.3,
            label="importance-weighted fit")
    ax.set_xlim(-1.2, 2.0)
    ax.set_ylim(-1.5, 3.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("(a) A linear model under covariate shift", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.0)
    ax = axes[1]
    ax.plot(xx, norm.pdf(xx, 0.5, 0.5), color=NAVY, label="$p_{\\mathrm{train}}(x)$")
    ax.plot(xx, norm.pdf(xx, 0.0, 0.3), color=GOLD, label="$p_{\\mathrm{deploy}}(x)$")
    ax2 = ax.twinx()
    order = np.argsort(m.x_s)
    ax2.plot(m.x_s[order], m.w_est[order], color=TEAL, lw=1.3, label="estimated weight")
    ax2.plot(m.x_s[order], m.w_true[order], color=TEAL, lw=1, ls="--", label="true weight")
    ax2.set_ylabel("importance weight $w(x)$", fontsize=7)
    ax2.tick_params(labelsize=6.5)
    ax2.grid(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.set_title("(b) Densities and weights", fontsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.0)
    save(fig, "covariate_shift")


def fig_transfer():
    m = quiet_import("l07_transfer")
    styles = {"target": (CORAL, "o-"), "source": (GRAY, "--"), "pooled": (NAVY, "s-"),
              "augmented": (GOLD, "^-"), "regularised": (TEAL, "D-")}
    fig, ax = plt.subplots(figsize=(HALF + 1.2, 2.5))
    for name, curve in m.curves.items():
        c, st = styles[name]
        ax.plot(m.sizes, curve, st, color=c, ms=3.5, lw=1.2, label=name)
    ax.set_xscale("log")
    ax.set_xticks(m.sizes)
    ax.set_xticklabels([str(s) for s in m.sizes])
    ax.set_xlabel("labelled examples of the target task")
    ax.set_ylabel("accuracy on the target task")
    ax.legend(loc="lower right", fontsize=6.3)
    save(fig, "transfer")


def fig_experimentation():
    ab = quiet_import("l08_ab_testing")
    bandits = quiet_import("l09_bandits")
    rng = np.random.default_rng(11)
    looks_list, rates = [1, 2, 5, 10, 20, 50, 100], []
    for looks in looks_list:
        seg = ab.n // looks
        users = seg * np.arange(1, looks + 1)
        hits = 0
        for _ in range(3000):
            ca = np.cumsum(rng.binomial(seg, 0.05, looks))
            cb = np.cumsum(rng.binomial(seg, 0.05, looks))
            hits += np.any(ab.z_test(ca, users, cb, users)[1] < 0.05)
        rates.append(hits / 3000)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45))
    fig.subplots_adjust(wspace=0.3)
    ax = axes[0]
    ax.plot(looks_list, rates, "o-", color=CORAL, ms=4)
    ax.axhline(0.05, color=GRAY, ls=":", lw=1)
    ax.set_ylim(0, 0.4)
    ax.text(12, 0.036, "nominal level 5%", fontsize=6.5, color=GRAY)
    ax.set_xscale("log")
    ax.set_xticks(looks_list)
    ax.set_xticklabels([str(v) for v in looks_list])
    ax.set_xlabel("times the results are checked")
    ax.set_ylabel("false positive rate (A/A tests)")
    ax.set_title("(a) The cost of peeking", fontsize=8)
    ax = axes[1]
    for (name, curve), c in zip(bandits.curves.items(), [NAVY, GOLD, TEAL, CORAL]):
        ax.plot(np.arange(1, len(curve) + 1), curve, color=c, lw=1.3, label=name)
    ax.set_xlabel("users")
    ax.set_ylabel("expected clicks lost (regret)")
    ax.set_title("(b) Bandits versus an A/B test", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.3)
    print("  peeking:", dict(zip(looks_list, np.round(rates, 3))))
    save(fig, "experimentation")


def fig_sequential():
    m = quiet_import("l14_sequential")
    rng = np.random.default_rng(3)
    looks = np.arange(1, m.K + 1)
    fig, ax = plt.subplots(figsize=(HALF + 1.2, 2.6))
    for drift, c in [(0.0, GRAY), (1.96 + 0.8416, TEAL)]:
        steps = rng.normal(drift / np.sqrt(m.K), 1.0, (12, m.K))
        z = np.cumsum(steps, axis=1) / np.sqrt(looks)
        for path in z:
            ax.plot(looks, path, color=c, lw=0.7, alpha=0.6)
    ax.plot(looks, m.designs["Pocock"], "s-", color=CORAL, lw=1.4, ms=3.5, label="Pocock")
    ax.plot(looks, -m.designs["Pocock"], "s-", color=CORAL, lw=1.0, ms=2.5, alpha=0.6)
    ax.plot(looks, -m.designs["O'Brien-Fleming"], "o-", color=NAVY, lw=1.0, ms=2.5, alpha=0.6)
    ax.plot([m.K], [-1.96], "D", color=GOLD, ms=4, alpha=0.6)
    ax.plot(looks, m.designs["O'Brien-Fleming"], "o-", color=NAVY, lw=1.4, ms=3.5,
            label="O'Brien-Fleming")
    ax.plot([m.K], [1.96], "D", color=GOLD, ms=5, label="single test at the end")
    ax.plot([], [], color=TEAL, lw=1, label="paths with a real effect")
    ax.plot([], [], color=GRAY, lw=1, label="paths without effect")
    ax.set_xticks(looks)
    ax.set_xlabel("look (equally spaced in sample size)")
    ax.set_ylabel("$z$ statistic")
    ax.set_ylim(-5, 6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.8, frameon=False)
    save(fig, "sequential")


def fig_bandits_advanced():
    ns = quiet_import("l17_nonstationary_bandits")
    ctx = quiet_import("l18_linucb")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45))
    fig.subplots_adjust(wspace=0.3)
    ax = axes[0]
    for (name, curve), c in zip(ns.curves.items(), [NAVY, TEAL]):
        ax.plot(np.arange(1, len(curve) + 1), curve, color=c, lw=1.3, label=name)
    ax.axvline(ns.SWITCH, color=CORAL, ls=":", lw=1)
    ax.text(ns.SWITCH + 400, 15, "the best variant\nchanges", fontsize=6.3, color=CORAL)
    ax.set_xlabel("users")
    ax.set_ylabel("expected clicks lost")
    ax.set_title("(a) A non-stationary problem", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.3)
    ax = axes[1]
    for (name, curve), c in zip(ctx.results.items(), [GRAY, NAVY, TEAL]):
        ax.plot(np.arange(1, len(curve) + 1), curve, color=c, lw=1.3, label=name)
    ax.set_xlabel("users")
    ax.set_ylabel("expected clicks lost")
    ax.set_title("(b) Using the context of each user", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.3)
    save(fig, "bandits_advanced")


ALL = [fig_shift_types, fig_drift_patterns, fig_concept_drift, fig_covariate_shift,
       fig_transfer, fig_experimentation, fig_sequential, fig_bandits_advanced]

if __name__ == "__main__":
    use_style()
    only = set(sys.argv[2:])
    for f in ALL:
        if not only or f.__name__ in only:
            print(f.__name__)
            f()
