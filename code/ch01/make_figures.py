"""Generate all the figures of Chapter 1 (Model evaluation).

Usage:
    python make_figures.py              # writes ../../figures/ch01/*.pdf
    python make_figures.py OUT_DIR      # writes OUT_DIR/*.pdf

Every function fig_xxx() produces one figure. All randomness is seeded, so the
figures (and the numbers quoted in the book) are reproducible.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, rankdata, friedmanchisquare, wilcoxon
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.datasets import (make_classification, make_moons, make_circles,
                              make_blobs, load_diabetes)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, silhouette_samples,
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, r2_score, mean_squared_error)
from sklearn.model_selection import (train_test_split, learning_curve,
                                     StratifiedKFold, cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from edm_style import (use_style, NAVY, BRONZE, TEAL, CORAL, GOLD, INDIGO,
                       GRAY, LIGHT, CYCLE, FULL, HALF)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "figures", "ch01")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"[saved] {name}.pdf")


def true_f(x):
    return np.sin(2 * np.pi * x)


# --------------------------------------------------------------------------
# 1.2 Bias and variance
# --------------------------------------------------------------------------
def fig_polyfits():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1, 20))
    y = true_f(x) + rng.normal(0, 0.25, 20)
    xt = rng.uniform(x.min(), x.max(), 5000)      # test points inside the training range
    yt = true_f(xt) + rng.normal(0, 0.25, 5000)
    xx = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.05), sharey=True)
    for ax, d, lab in zip(axes, [1, 4, 15], ["underfitting", "good fit", "overfitting"]):
        p = np.polynomial.Polynomial.fit(x, y, d)
        tr, te = np.mean((p(x) - y) ** 2), np.mean((p(xt) - yt) ** 2)
        ax.plot(xx, true_f(xx), "--", color=GRAY, lw=1.1, label="true $f(x)$")
        ax.plot(xx, p(xx), color=NAVY, label="fitted model")
        ax.scatter(x, y, s=13, color=CORAL, zorder=3, label="training data")
        ax.set_ylim(-1.9, 1.9)
        ax.set_title(f"Degree {d}: {lab}")
        ax.text(0.03, 0.04, f"train MSE = {tr:.3f}\ntest MSE  = {te:.3g}",
                transform=ax.transAxes, fontsize=6.8, va="bottom")
        ax.set_xlabel("$x$")
        print(f"  degree {d}: train MSE {tr:.4f}  test MSE {te:.4g}")
    axes[0].set_ylabel("$y$")
    axes[0].legend(loc="upper right", fontsize=6.3)
    save(fig, "polyfits")


def fig_bias_variance():
    """Fixed-design simulation: the inputs x are fixed, the noise is redrawn R times."""
    rng = np.random.default_rng(2)
    sigma, n, R = 0.25, 40, 500
    degrees = np.arange(0, 16)
    x = np.sort(rng.uniform(0, 1, n))
    x0 = np.linspace(x.min(), x.max(), 80)
    preds = np.zeros((len(degrees), R, len(x0)))
    tr_err = np.zeros((len(degrees), R))
    te_err = np.zeros((len(degrees), R))
    for r in range(R):
        y = true_f(x) + rng.normal(0, sigma, n)
        xt = rng.uniform(x.min(), x.max(), 400)
        yt = true_f(xt) + rng.normal(0, sigma, 400)
        for i, d in enumerate(degrees):
            p = np.polynomial.Polynomial.fit(x, y, d)
            preds[i, r] = p(x0)
            tr_err[i, r] = np.mean((p(x) - y) ** 2)
            te_err[i, r] = np.mean((p(xt) - yt) ** 2)
    bias2 = np.mean((preds.mean(axis=1) - true_f(x0)) ** 2, axis=1)
    var = np.mean(preds.var(axis=1), axis=1)
    total = bias2 + var + sigma ** 2
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35))
    ax = axes[0]
    ax.plot(degrees, tr_err.mean(1), "o-", ms=3, color=TEAL, label="training error")
    ax.plot(degrees, te_err.mean(1), "o-", ms=3, color=CORAL, label="test error")
    ax.axhline(sigma ** 2, ls=":", color=GRAY, lw=1)
    ax.text(9.5, sigma ** 2 - 0.014, r"noise $\sigma^2$", fontsize=6.8, color=GRAY)
    ax.set_ylim(0, 0.25)
    ax.text(0.4, 0.232, "degree 0: 0.59 (off scale)", fontsize=6.3, color=GRAY)
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("mean squared error")
    ax.set_title("(a) Training vs. test error")
    ax.legend(loc="center right")
    ax = axes[1]
    ax.plot(degrees, bias2, "o-", ms=3, color=NAVY, label="bias$^2$")
    ax.plot(degrees, var, "o-", ms=3, color=GOLD, label="variance")
    ax.axhline(sigma ** 2, ls=":", color=GRAY, lw=1, label=r"noise $\sigma^2$")
    ax.plot(degrees, total, "-", color=CORAL, lw=2, label="expected test error")
    best = degrees[np.argmin(total)]
    ax.axvline(best, color=CORAL, ls="--", lw=0.8)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("polynomial degree")
    ax.set_title("(b) Bias--variance decomposition")
    ax.legend(loc="upper right", fontsize=6.6)
    print(f"  best degree (min expected error) = {best}; bias2={np.round(bias2, 4)}; var={np.round(var, 4)}")
    save(fig, "bias_variance")


def fig_knn():
    X, y = make_moons(n_samples=300, noise=0.30, random_state=0)
    Xt, yt = make_moons(n_samples=4000, noise=0.30, random_state=1)
    xx, yy = np.meshgrid(np.linspace(-1.7, 2.7, 300), np.linspace(-1.3, 1.8, 220))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.95), sharey=True)
    for ax, k in zip(axes, [1, 15, 150]):
        m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        Z = m.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=["#DCE3E7", "#F7DDD6"])
        ax.contour(xx, yy, Z, levels=[0.5], colors=[NAVY], linewidths=0.9)
        ax.scatter(*X[y == 0].T, s=6, color=NAVY)
        ax.scatter(*X[y == 1].T, s=6, color=CORAL)
        tr, te = m.score(X, y), m.score(Xt, yt)
        ax.set_title(f"$k={k}$\ntrain acc. {tr:.2f}, test acc. {te:.2f}", fontsize=7.6)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  k={k}: train {tr:.3f} test {te:.3f}")
    save(fig, "knn_boundaries")


def fig_learning_curves():
    X, y = make_moons(n_samples=3000, noise=0.3, random_state=3)
    sizes = np.linspace(0.02, 1.0, 12)
    models = [("Logistic regression\n(high bias)", LogisticRegression()),
              ("Unpruned decision tree\n(high variance)", DecisionTreeClassifier(random_state=0)),
              ("$k$-NN with $k=25$\n(good compromise)", KNeighborsClassifier(25))]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), sharey=True)
    for ax, (name, m) in zip(axes, models):
        n, tr, va = learning_curve(m, X, y, train_sizes=sizes, cv=cv, scoring="accuracy")
        for s, c, lab in [(tr, TEAL, "training"), (va, CORAL, "validation (CV)")]:
            ax.plot(n, s.mean(1), "o-", ms=2.5, color=c, label=lab)
            ax.fill_between(n, s.mean(1) - s.std(1), s.mean(1) + s.std(1),
                            color=c, alpha=0.15, lw=0)
        ax.set_title(name, fontsize=7.8)
        ax.set_xlabel("training set size")
        ax.set_ylim(0.74, 1.01)
        print(f"  {name.splitlines()[0]}: final train {tr.mean(1)[-1]:.3f} val {va.mean(1)[-1]:.3f}")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="lower right")
    save(fig, "learning_curves")


# --------------------------------------------------------------------------
# 1.4 Crisp classifiers
# --------------------------------------------------------------------------
def fig_prevalence():
    prev = np.logspace(-3, np.log10(0.5), 400)
    fig, ax = plt.subplots(figsize=(HALF + 0.9, 2.45))
    tests = [((0.95, 0.98), NAVY, "test A: Se = 0.95, Sp = 0.98"),
             ((0.60, 0.995), CORAL, "test B: Se = 0.60, Sp = 0.995")]
    for (se, sp), c, lab in tests:
        ppv = se * prev / (se * prev + (1 - sp) * (1 - prev))
        npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev)
        ax.plot(prev, ppv, color=c, label=f"PPV, {lab}")
        ax.plot(prev, npv, "--", color=c, lw=1.1, label=f"NPV, {lab}")
        p1 = 0.01
        v = se * p1 / (se * p1 + (1 - sp) * (1 - p1))
        ax.plot(p1, v, "o", color=c, ms=4)
        print(f"  {lab}: PPV at 1% = {v:.3f}")
    ax.axvline(0.01, color=GRAY, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.001, 0.01, 0.1, 0.5])
    ax.set_xticklabels(["0.1%", "1%", "10%", "50%"])
    ax.set_xlabel("prevalence $P/(P+N)$")
    ax.set_ylabel("predictive value")
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", fontsize=6.3)
    save(fig, "prevalence_ppv")


# --------------------------------------------------------------------------
# 1.5 ROC analysis
# --------------------------------------------------------------------------
ROC_PTS = {"CN2": (0.05, 0.50), "Ripper": (0.10, 0.60), "C4.5": (0.16, 0.80),
           "NB": (0.28, 0.70), "SVM": (0.60, 0.95)}


def fig_signal_detection():
    s = np.linspace(-4, 6, 800)
    t, mu = 1.0, 1.5
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    ax.plot(s, norm.pdf(s), color=NAVY, label="negatives (noise)")
    ax.plot(s, norm.pdf(s, mu), color=CORAL, label="positives (signal + noise)")
    r = s >= t
    ax.fill_between(s[r], norm.pdf(s[r], mu), color=CORAL, alpha=0.22, lw=0,
                    label=f"TPR $=P(s>t\\mid +)$ = {norm.sf(t, mu):.2f}")
    ax.fill_between(s[r], norm.pdf(s[r]), color=NAVY, alpha=0.45, lw=0,
                    label=f"FPR $=P(s>t\\mid -)$ = {norm.sf(t):.2f}")
    ax.axvline(t, color=GRAY, ls="--", lw=1)
    ax.text(t + 0.08, 0.425, "threshold $t$", fontsize=7)
    ax.set_xlabel("score $s(x)$")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.66)
    ax.legend(loc="upper left", fontsize=6.3)
    ax.set_title("(a) Two score distributions and a threshold")
    ax = axes[1]
    tt = np.linspace(-7, 10, 600)
    for d, c in [(0.5, GOLD), (1.5, CORAL), (3.0, TEAL)]:
        ax.plot(norm.sf(tt), norm.sf(tt, d), color=c, label=f"$d'={d}$")
    ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1)
    ax.plot(norm.sf(t), norm.sf(t, mu), "o", color=CORAL, ms=5, mec="white")
    ax.annotate("$t=1$", (norm.sf(t), norm.sf(t, mu)), xytext=(0.32, 0.52),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("(b) ROC curves")
    save(fig, "signal_detection")


def fig_roc_space():
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.fill_between([0, 1], [0, 1], [1, 1], color=TEAL, alpha=0.07, lw=0)
    ax.fill_between([0, 1], [0, 0], [0, 1], color=CORAL, alpha=0.07, lw=0)
    ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1.2)
    special = [((0, 1), "ROC heaven\n(perfect)", (0.04, 0.88), "left"),
               ((1, 1), "AlwaysPos", (0.96, 0.92), "right"),
               ((0, 0), "AlwaysNeg", (0.04, 0.05), "left"),
               ((1, 0), "ROC hell", (0.96, 0.05), "right"),
               ((0.5, 0.5), "random\n($p=0.5$)", (0.53, 0.40), "left")]
    for (x, y), lab, (tx, ty), ha in special:
        ax.plot(x, y, "o", color=INDIGO, ms=5, clip_on=False, zorder=4)
        ax.text(tx, ty, lab, fontsize=7, ha=ha)
    ax.plot(0.30, 0.75, "o", color=CORAL, ms=6, zorder=4)
    ax.text(0.33, 0.76, "A (0.30, 0.75)", fontsize=7)
    ax.plot(0.70, 0.25, "s", color=CORAL, ms=5, mfc="white", zorder=4)
    ax.text(0.53, 0.18, "A' (0.70, 0.25)", fontsize=7)
    ax.annotate("", xy=(0.31, 0.72), xytext=(0.69, 0.28),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1, ls="--"))
    ax.text(0.58, 0.63, "invert\npredictions", fontsize=6.5, color=CORAL)
    ax.text(0.08, 0.60, "better than\nrandom", fontsize=6.8, color=TEAL)
    ax.text(0.76, 0.50, "worse than\nrandom", fontsize=6.8, color=CORAL)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate (FPR)")
    ax.set_ylabel("true positive rate (TPR)")
    save(fig, "roc_space")


def _roc_points(ax):
    ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1)
    off = {"CN2": (0.025, -0.03), "Ripper": (0.025, -0.035), "C4.5": (-0.02, 0.035),
           "NB": (0.025, -0.035), "SVM": (-0.02, 0.03)}
    for name, (x, y) in ROC_PTS.items():
        ax.plot(x, y, "o", color=NAVY, ms=4.5, zorder=4)
        dx, dy = off[name]
        ax.text(x + dx, y + dy, name, fontsize=7, ha="left" if dx > 0 else "right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate")


def fig_roc_hull():
    hull = np.array([(0, 0), ROC_PTS["CN2"], ROC_PTS["C4.5"], ROC_PTS["SVM"], (1, 1)])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.95))
    for ax in axes:
        _roc_points(ax)
        ax.plot(hull[:, 0], hull[:, 1], "-", color=TEAL, lw=1.6, zorder=3)
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("(a) Five classifiers and their ROC convex hull")
    ax = axes[1]
    xs = np.linspace(-0.2, 1.2, 10)
    for C, name, c, lab in [(1.0, "C4.5", NAVY, "$C=1$ (balanced)"),
                            (0.25, "SVM", CORAL, "$C=1/4$ (4 pos : 1 neg)"),
                            (4.0, "CN2", GOLD, "$C=4$ (1 pos : 4 neg)")]:
        x0, y0 = ROC_PTS[name]
        ax.plot(xs, y0 + C * (xs - x0), "--", color=c, lw=1.2, label=lab)
    ax.legend(loc="lower right", fontsize=6.6)
    ax.set_title("(b) Iso-accuracy lines touching the hull")
    save(fig, "roc_hull")


def fig_roc_construction():
    y = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    s = np.array([.95, .90, .85, .80, .70, .70, .60, .55, .40, .20])
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.fill_between(fpr, tpr, color=TEAL, alpha=0.14, lw=0)
    ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1)
    ax.plot(fpr, tpr, "-o", color=NAVY, ms=3.5, lw=1.4)
    offsets = {0.95: (5, -3), 0.90: (5, 2), 0.85: (3, -11), 0.80: (-20, 0), 0.70: (-20, 3),
               0.60: (2, -11), 0.55: (3, -11), 0.40: (-8, -11), 0.20: (-18, -11)}
    for t_, x_, y_ in zip(thr[1:], fpr[1:], tpr[1:]):
        ax.annotate(f"{t_:.2f}", (x_, y_), xytext=offsets[round(t_, 2)], textcoords="offset points",
                    fontsize=6.3, color=CORAL)
    ax.text(0.30, 0.62, "tie (0.70):\ndiagonal step", fontsize=6.3, color=GRAY)
    auc = roc_auc_score(y, s)
    ax.text(0.55, 0.32, f"AUC = {auc:.2f}", fontsize=8.5, color=TEAL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    print(f"  AUC ranking example = {auc:.3f}")
    save(fig, "roc_construction")


def _synthetic_binary(random_state=11, n=4000):
    X, y = make_classification(n_samples=n, n_features=20, n_informative=6,
                               n_redundant=4, flip_y=0.05, class_sep=0.8,
                               random_state=random_state)
    return train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)


def fig_roc_models():
    X_tr, X_te, y_tr, y_te = _synthetic_binary()
    models = {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
              "Naive Bayes": GaussianNB(),
              "$k$-NN ($k=15$)": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
              "Random forest": RandomForestClassifier(300, random_state=0)}
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1)
    for (name, m), c in zip(models.items(), [NAVY, GOLD, TEAL, CORAL]):
        p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        auc = roc_auc_score(y_te, p)
        ax.plot(fpr, tpr, color=c, lw=1.3, label=f"{name} (AUC = {auc:.3f})")
        yhat = p >= 0.5
        ax.plot(yhat[y_te == 0].mean(), yhat[y_te == 1].mean(), "o", color=c, ms=5,
                mec="white", zorder=5)
        print(f"  {name}: AUC {auc:.3f} acc {np.mean(yhat == y_te):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right", fontsize=6.3)
    save(fig, "roc_models")


def _two_gaussians(npos, nneg, rng, mu=1.5):
    s = np.r_[rng.normal(mu, 1, npos), rng.normal(0, 1, nneg)]
    y = np.r_[np.ones(npos), np.zeros(nneg)]
    return y, s


def fig_gains_lift():
    rng = np.random.default_rng(8)
    y, s = _two_gaussians(500, 9500, rng, mu=1.6)
    ys = y[np.argsort(-s)]
    frac = np.arange(1, len(y) + 1) / len(y)
    gain = np.cumsum(ys) / ys.sum()
    i20 = int(0.2 * len(y)) - 1
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    ax = axes[0]
    ax.plot(frac, gain, color=NAVY, label="model")
    ax.plot([0, 1], [0, 1], ":", color=GRAY, label="random targeting")
    ax.plot([0, ys.mean(), 1], [0, 1, 1], "--", color=TEAL, lw=1.1, label="perfect model")
    ax.plot(0.2, gain[i20], "o", color=CORAL, ms=4)
    ax.annotate(f"top 20% contains\n{100 * gain[i20]:.0f}% of responders", (0.2, gain[i20]),
                xytext=(0.33, 0.35), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("fraction contacted (ranked by score)")
    ax.set_ylabel("fraction of responders reached")
    ax.set_title("(a) Cumulative gains chart")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(frac, gain / frac, color=CORAL)
    ax.axhline(1, ls=":", color=GRAY)
    ax.plot(0.2, gain[i20] / 0.2, "o", color=CORAL, ms=4)
    ax.annotate(f"lift at 20% = {gain[i20] / 0.2:.2f}", (0.2, gain[i20] / 0.2),
                xytext=(0.35, 3.7), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_ylim(0, 8)
    ax.set_xlabel("fraction contacted (ranked by score)")
    ax.set_ylabel("lift")
    ax.set_title("(b) Lift chart")
    print(f"  gains at 20% = {gain[i20]:.3f}, lift = {gain[i20] / 0.2:.2f}, AUC = {roc_auc_score(y, s):.3f}")
    save(fig, "gains_lift")


# --------------------------------------------------------------------------
# 1.7 Calibration
# --------------------------------------------------------------------------
class MinMaxSVC:
    """Linear SVM whose decision values are naively min-max scaled to [0, 1]."""

    def __init__(self):
        self.svc = LinearSVC(C=1.0, max_iter=20000)

    def fit(self, X, y):
        self.svc.fit(X, y)
        d = self.svc.decision_function(X)
        self.lo, self.hi = d.min(), d.max()
        return self

    def predict_proba(self, X):
        p = np.clip((self.svc.decision_function(X) - self.lo) / (self.hi - self.lo), 0, 1)
        return np.c_[1 - p, p]


def ece(y, p, n_bins=10):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
               for b in range(n_bins) if (bins == b).any())


def fig_reliability():
    X, y = make_classification(n_samples=100_000, n_features=20, n_informative=2,
                               n_redundant=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=1000, random_state=42)
    models = [("Logistic regression", LogisticRegression(), NAVY),
              ("Naive Bayes", GaussianNB(), GOLD),
              ("SVM (scaled scores)", MinMaxSVC(), TEAL),
              ("Random forest", RandomForestClassifier(200, random_state=0), CORAL)]
    fig = plt.figure(figsize=(FULL, 3.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1.25, 1, 1], wspace=0.45, hspace=0.6)
    axc = fig.add_subplot(gs[:, :2])
    axc.plot([0, 1], [0, 1], ":", color=GRAY, lw=1, label="perfectly calibrated")
    for i, (name, m, c) in enumerate(models):
        p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        frac, mean_p = calibration_curve(y_te, p, n_bins=10)
        axc.plot(mean_p, frac, "o-", ms=3, color=c, label=name)
        axh = fig.add_subplot(gs[i // 2, 2 + i % 2])
        axh.hist(p, bins=20, range=(0, 1), color=c, alpha=0.85)
        axh.set_title(name, fontsize=6.8)
        axh.set_yticks([])
        axh.tick_params(labelsize=6.3)
        axh.grid(False)
        print(f"  {name}: ECE {ece(y_te, p):.4f}")
    axc.set_xlabel("mean predicted probability (bin)")
    axc.set_ylabel("observed fraction of positives (bin)")
    axc.set_aspect("equal")
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.legend(loc="upper left", fontsize=6.3)
    save(fig, "reliability")


def fig_calibration_methods():
    X, y = make_classification(n_samples=40_000, n_features=20, n_informative=2,
                               n_redundant=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    svc = lambda: LinearSVC(C=1.0, max_iter=20000)
    models = [("SVM, min-max scaled scores", MinMaxSVC(), GOLD),
              ("+ Platt scaling", CalibratedClassifierCV(svc(), method="sigmoid", cv=5), NAVY),
              ("+ isotonic regression", CalibratedClassifierCV(svc(), method="isotonic", cv=5), CORAL)]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot([0, 1], [0, 1], ":", color=GRAY, lw=1)
    for name, m, c in models:
        p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        frac, mean_p = calibration_curve(y_te, p, n_bins=10)
        e = ece(y_te, p)
        ax.plot(mean_p, frac, "o-", ms=3, color=c, label=f"{name} (ECE = {e:.3f})")
        print(f"  {name}: ECE {e:.4f}")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed fraction of positives")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.3)
    save(fig, "calibration_methods")


# --------------------------------------------------------------------------
# 1.8 Imbalanced classification
# --------------------------------------------------------------------------
def fig_pr_vs_roc():
    rng = np.random.default_rng(5)
    cases = [("50% positives", 5000, 5000, NAVY), ("2% positives", 200, 9800, CORAL)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.75))
    for name, npos, nneg, c in cases:
        y, s = _two_gaussians(npos, nneg, rng)
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
        axes[0].plot(fpr, tpr, color=c, lw=1.3, label=f"{name}: AUC = {auc:.3f}")
        axes[1].plot(rec, prec, color=c, lw=1.3, label=f"{name}: AP = {ap:.3f}")
        axes[1].axhline(npos / (npos + nneg), ls=":", color=c, lw=1)
        print(f"  {name}: AUC {auc:.3f}  AP {ap:.3f}")
    axes[0].plot([0, 1], [0, 1], ":", color=GRAY, lw=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("(a) ROC curves (almost identical)")
    axes[1].set_xlabel("recall (TPR)")
    axes[1].set_ylabel("precision (PPV)")
    axes[1].set_title("(b) Precision--recall curves (very different)")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes[0].legend(loc="lower right", fontsize=6.6)
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.27), fontsize=6.6)
    save(fig, "pr_vs_roc")


def smote(X_min, n_new, k=5, seed=0):
    """Generate n_new synthetic minority examples (Chawla et al., 2002)."""
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    base = rng.integers(0, len(X_min), n_new)
    neigh = idx[base, rng.integers(1, k + 1, n_new)]
    gap = rng.random((n_new, 1))
    return X_min[base] + gap * (X_min[neigh] - X_min[base])


def fig_smote():
    rng = np.random.default_rng(4)
    X_maj = rng.multivariate_normal([0, 0], [[1.2, 0.5], [0.5, 1.0]], 400)
    X_min = rng.multivariate_normal([2.0, 1.6], [[0.35, 0.1], [0.1, 0.35]], 30)
    X_syn = smote(X_min, 150, k=5, seed=1)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharex=True, sharey=True)
    for ax in axes:
        ax.scatter(*X_maj.T, s=6, color=LIGHT, edgecolor="#A9A9A9", lw=0.3, label="majority")
        ax.scatter(*X_min.T, s=14, color=CORAL, zorder=3, label="minority")
    # one SMOTE step
    nn = NearestNeighbors(n_neighbors=6).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    i = 3
    for j in idx[i, 1:]:
        axes[0].plot(*np.c_[X_min[i], X_min[j]], "--", color=NAVY, lw=0.8, zorder=2)
    j = idx[i, 2]
    new = X_min[i] + 0.4 * (X_min[j] - X_min[i])
    axes[0].plot(*X_min[i], "o", ms=9, mfc="none", mec=NAVY, mew=1.2, zorder=4)
    axes[0].plot(*new, "*", ms=11, color=GOLD, mec=NAVY, mew=0.6, zorder=5, label="synthetic")
    axes[0].set_title("(a) One SMOTE step: base point, its $k=5$\nneighbours and a new point on a segment",
                      fontsize=7.5)
    axes[1].scatter(*X_syn.T, s=10, marker="x", color=GOLD, lw=0.9, zorder=4, label="synthetic")
    axes[1].set_title("(b) After SMOTE: 150 synthetic minority examples", fontsize=7.5)
    axes[0].legend(loc="lower right", fontsize=6.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, "smote")


# --------------------------------------------------------------------------
# 1.9 Costs
# --------------------------------------------------------------------------
def fig_cost_threshold():
    X, y = make_classification(n_samples=60_000, n_features=10, n_informative=5,
                               weights=[0.9], flip_y=0.02, class_sep=1.3, random_state=21)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
    p = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    t = np.linspace(0.005, 0.995, 199)
    fig, ax = plt.subplots(figsize=(HALF + 0.8, 2.5))
    for ratio, c in [(1, NAVY), (5, TEAL), (20, CORAL)]:
        cost = np.array([(np.sum((p >= ti) & (y_te == 0)) + ratio * np.sum((p < ti) & (y_te == 1)))
                         / len(y_te) for ti in t])
        ax.plot(t, cost, color=c, label=f"$c_{{FN}}/c_{{FP}} = {ratio}$")
        tstar = 1 / (1 + ratio)
        ax.axvline(tstar, color=c, ls="--", lw=0.9)
        k = np.argmin(cost)
        ax.plot(t[k], cost[k], "o", color=c, ms=4)
        print(f"  ratio {ratio}: t* = {tstar:.3f}, empirical best t = {t[k]:.3f}")
    ax.set_xlabel("decision threshold $t$")
    ax.set_ylabel("average cost per example")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right")
    save(fig, "cost_threshold")


def fig_cost_curves():
    pc = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    curves = []
    for (name, (fpr, tpr)), c in zip(ROC_PTS.items(), [GOLD, INDIGO, NAVY, BRONZE, CORAL]):
        nec = (1 - tpr) * pc + fpr * (1 - pc)
        curves.append(nec)
        ax.plot(pc, nec, color=c, lw=1.0, label=name)
    ax.plot(pc, pc, ":", color=GRAY, lw=1)
    ax.plot(pc, 1 - pc, ":", color=GRAY, lw=1)
    ax.text(0.385, 0.565, "AlwaysPos", fontsize=6.5, color=GRAY, ha="right")
    ax.text(0.615, 0.565, "AlwaysNeg", fontsize=6.5, color=GRAY, ha="left")
    env = np.min(np.vstack(curves + [pc, 1 - pc]), axis=0)
    ax.plot(pc, env, color=TEAL, lw=2.4, alpha=0.6, label="lower envelope")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("probability cost $PC(+)$")
    ax.set_ylabel("normalised expected cost")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.6)
    save(fig, "cost_curves")


# --------------------------------------------------------------------------
# 1.10 Regression
# --------------------------------------------------------------------------
def fig_regression_diagnostics():
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                  random_state=0).fit(X_tr, y_tr)
    pred = m.predict(X_te)
    res = y_te - pred
    r2, rmse = r2_score(y_te, pred), np.sqrt(mean_squared_error(y_te, pred))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5))
    ax = axes[0]
    ax.scatter(y_te, pred, s=9, color=NAVY, alpha=0.75)
    lim = [20, 350]
    ax.plot(lim, lim, "--", color=CORAL, lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual value $y$")
    ax.set_ylabel(r"prediction $\hat y$")
    ax.set_title(f"(a) Predicted vs. actual ($R^2$ = {r2:.2f}, RMSE = {rmse:.1f})")
    ax = axes[1]
    ax.scatter(pred, res, s=9, color=TEAL, alpha=0.75)
    ax.axhline(0, color=CORAL, ls="--", lw=1)
    ax.set_xlabel(r"prediction $\hat y$")
    ax.set_ylabel(r"residual $y-\hat y$")
    ax.set_title("(b) Residuals vs. predictions")
    print(f"  diabetes GBR: R2 {r2:.3f} RMSE {rmse:.2f}")
    save(fig, "regression_diagnostics")


def fig_regression_losses():
    e = np.linspace(-3, 3, 601)          # e = prediction - actual
    a, tau, delta = -0.8, 0.8, 1.0
    huber = np.where(np.abs(e) <= delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))
    linex = np.exp(a * e) - a * e - 1
    pinball = np.where(e <= 0, -tau * e, (1 - tau) * e)
    fig, ax = plt.subplots(figsize=(HALF + 0.8, 2.5))
    ax.plot(e, e ** 2, color=NAVY, label="squared $e^2$")
    ax.plot(e, np.abs(e), color=TEAL, label="absolute $|e|$")
    ax.plot(e, huber, color=INDIGO, ls="--", lw=1.2, label=r"Huber ($\delta=1$)")
    ax.plot(e, linex, color=CORAL, label=r"LinEx ($a=-0.8$)")
    ax.plot(e, pinball, color=GOLD, label=r"pinball ($\tau=0.8$)")
    ax.set_ylim(0, 6)
    ax.set_xlabel(r"error $e=\hat y - y$  ($e<0$: under-prediction)")
    ax.set_ylabel("loss")
    ax.legend(loc="upper center", ncol=2, fontsize=6.6)
    save(fig, "regression_losses")


# --------------------------------------------------------------------------
# 1.11 Other supervised settings
# --------------------------------------------------------------------------
def fig_conformal():
    rng = np.random.default_rng(2)
    n, alpha = 8000, 0.1
    X = rng.uniform(0, 10, (n, 1))
    y = X[:, 0] * np.sin(X[:, 0]) + rng.normal(0, 0.3 + 0.3 * X[:, 0])
    idx = rng.permutation(n)
    tr, cal, te = idx[:2000], idx[2000:4000], idx[4000:]
    level = np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal)
    m = GradientBoostingRegressor(random_state=0).fit(X[tr], y[tr])
    q = np.quantile(np.abs(y[cal] - m.predict(X[cal])), level, method="higher")
    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=0).fit(X[tr], y[tr])
    hi = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, random_state=0).fit(X[tr], y[tr])
    E = np.maximum(lo.predict(X[cal]) - y[cal], y[cal] - hi.predict(X[cal]))
    qc = np.quantile(E, level, method="higher")
    xx = np.linspace(0, 10, 400)[:, None]
    bands = [("(a) Split conformal (absolute residuals)", m.predict(xx) - q, m.predict(xx) + q,
              m.predict(X[te]) - q, m.predict(X[te]) + q),
             ("(b) Conformalised quantile regression", lo.predict(xx) - qc, hi.predict(xx) + qc,
              lo.predict(X[te]) - qc, hi.predict(X[te]) + qc)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45), sharey=True)
    sub = te[:600]
    xs = X[te, 0]
    for ax, (title, l, u, lt, ut) in zip(axes, bands):
        inside = (y[te] >= lt) & (y[te] <= ut)
        cov, width = inside.mean(), np.mean(ut - lt)
        ax.fill_between(xx[:, 0], l, u, color=TEAL, alpha=0.2, lw=0, label="90% prediction band")
        ax.scatter(X[sub, 0], y[sub], s=4, color=NAVY, alpha=0.6, label="test data")
        ax.set_title(f"{title}\ncoverage = {cov:.3f}, mean width = {width:.2f}", fontsize=7.6)
        ax.set_xlabel("$x$")
        print(f"  {title}: coverage {cov:.3f} width {width:.2f} | "
              f"coverage x<3: {inside[xs < 3].mean():.3f}, x>7: {inside[xs > 7].mean():.3f}")
    axes[0].set_ylabel("$y$")
    axes[0].legend(loc="lower left", fontsize=6.5)
    save(fig, "conformal")


def fig_open_set():
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [4, 0], [2, 3.4]])
    Xk = np.vstack([rng.normal(c, 0.8, (400, 2)) for c in centers])
    yk = np.repeat([0, 1, 2], 400)
    Xk_tr, Xk_te, yk_tr, yk_te = train_test_split(Xk, yk, test_size=0.5, random_state=0, stratify=yk)
    unk_centers = np.array([[2, -3.6], [6.6, 3.4], [-2.6, 3.4]])
    Xu = np.vstack([rng.normal(c, 0.8, (150, 2)) for c in unk_centers])
    clf = LogisticRegression(max_iter=1000).fit(Xk_tr, yk_tr)
    pred = clf.predict(Xk_te)
    means = np.array([Xk_tr[yk_tr == c].mean(0) for c in range(3)])

    def dist_score(Z):
        return -np.min(np.linalg.norm(Z[:, None, :] - means[None], axis=2), axis=1)

    scores = {"max. softmax probability": (clf.predict_proba(Xk_te).max(1), clf.predict_proba(Xu).max(1), CORAL),
              "distance to class mean": (dist_score(Xk_te), dist_score(Xu), TEAL)}
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.25))
    fig.subplots_adjust(wspace=0.38)
    ax = axes[0]
    xx, yy = np.meshgrid(np.linspace(-5, 9, 250), np.linspace(-6, 6.5, 250))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#DCE3E7", "#F7DDD6", "#D5EAE7"])
    for c, col in zip(range(3), [NAVY, CORAL, TEAL]):
        ax.scatter(*Xk_te[yk_te == c].T, s=3, color=col)
    ax.scatter(*Xu.T, s=6, marker="x", color="black", lw=0.6, label="unknown")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("(a) Data", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=6.2)
    ax = axes[1]
    sk, su, _ = scores["max. softmax probability"]
    ax.hist(sk, bins=28, range=(0.3, 1), color=NAVY, alpha=0.6, density=True, label="known")
    ax.hist(su, bins=28, range=(0.3, 1), color=CORAL, alpha=0.6, density=True, label="unknown")
    ax.set_yscale("log")
    ax.set_xlabel("max. softmax probability")
    ax.set_ylabel("density (log scale)")
    ax.set_title("(b) Classifier confidence", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=6.2)
    ax = axes[2]
    correct = pred == yk_te
    for name, (sk, su, col) in scores.items():
        ths = np.unique(np.r_[sk, su])[::-1]
        ccr = np.array([np.mean(correct & (sk >= t)) for t in ths])
        fpr = np.array([np.mean(su >= t) for t in ths])
        auroc = roc_auc_score(np.r_[np.ones(len(sk)), np.zeros(len(su))], np.r_[sk, su])
        ax.plot(fpr, ccr, color=col, lw=1.3, label=f"{name}\n(AUROC = {auroc:.3f})")
        print(f"  open set {name}: AUROC {auroc:.3f}")
    ax.set_xlabel("FPR (unknowns accepted)")
    ax.set_ylabel("CCR")
    ax.set_title("(c) OSCR curves", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", fontsize=5.8)
    print(f"  closed-set accuracy {correct.mean():.3f}")
    save(fig, "open_set")


# --------------------------------------------------------------------------
# 1.12 Computer vision
# --------------------------------------------------------------------------
def fig_detection_pr():
    tp = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    n_gt = 5
    prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    rec = np.cumsum(tp) / n_gt
    env = np.maximum.accumulate(prec[::-1])[::-1]
    fig, ax = plt.subplots(figsize=(HALF, 2.5))
    levels = np.unique(rec)
    prev = 0.0
    ap = 0.0
    for r in levels:
        h = env[rec == r].max()
        ax.add_patch(plt.Rectangle((prev, 0), r - prev, h, color=TEAL, alpha=0.15, lw=0))
        ax.plot([prev, r], [h, h], color=TEAL, lw=1.5)
        ap += (r - prev) * h
        prev = r
    ax.plot(rec, prec, ":", color=CORAL, lw=1)
    ax.plot(rec, prec, "o", color=CORAL, ms=4, label="raw (recall, precision)")
    for i, (r, p_) in enumerate(zip(rec, prec), 1):
        ax.text(r + 0.012, p_ + 0.02, str(i), fontsize=6.3)
    ax.plot([], [], color=TEAL, lw=1.5, label="interpolated precision")
    ax.text(0.05, 0.3, f"AP = {ap:.3f}", fontsize=8.5, color=TEAL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(loc="lower right", fontsize=6.4)
    print(f"  detection AP = {ap:.4f}")
    save(fig, "detection_pr")


def fig_dice_iou():
    j = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.plot(j, 2 * j / (1 + j), color=NAVY, label="Dice $=2J/(1+J)$")
    ax.plot(j, j, ":", color=GRAY, label="Dice $=J$")
    ax.plot(0.231, 0.375, "o", color=CORAL, ms=4)
    ax.annotate("boxes of the\nworked example", (0.231, 0.375), xytext=(0.42, 0.2), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("IoU (Jaccard index) $J$")
    ax.set_ylabel("Dice coefficient")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=6.5)
    save(fig, "dice_iou")


# --------------------------------------------------------------------------
# 1.14 Unsupervised
# --------------------------------------------------------------------------
def _blobs():
    return make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=7)


def fig_silhouette():
    X, _ = _blobs()
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.0), gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(hspace=0.42)
    for j, k in enumerate([3, 4, 5]):
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        sil = silhouette_samples(X, lab)
        avg = sil.mean()
        ax = axes[0, j]
        y0 = 10
        for c in range(k):
            v = np.sort(sil[lab == c])
            ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=CYCLE[c], alpha=0.85, lw=0)
            ax.text(-0.09, y0 + len(v) / 2, str(c + 1), fontsize=6.5, va="center")
            y0 += len(v) + 10
        ax.axvline(avg, color="black", ls="--", lw=0.9)
        ax.set_xlim(-0.15, 1)
        ax.set_yticks([])
        ax.set_xlabel("silhouette $s(i)$")
        ax.set_title(f"$k={k}$ (mean silhouette {avg:.3f})", fontsize=7.6)
        axes[1, j].scatter(*X.T, c=[CYCLE[l] for l in lab], s=4)
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        axes[1, j].grid(False)
        print(f"  k={k}: mean silhouette {avg:.3f}")
    save(fig, "silhouette")


def fig_internal_indices():
    X, y = _blobs()
    ks = np.arange(2, 9)
    res = {"silhouette (higher is better)": [], "Calinski-Harabasz (higher is better)": [],
           "Davies-Bouldin (lower is better)": []}
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        res["silhouette (higher is better)"].append(silhouette_score(X, lab))
        res["Calinski-Harabasz (higher is better)"].append(calinski_harabasz_score(X, lab))
        res["Davies-Bouldin (lower is better)"].append(davies_bouldin_score(X, lab))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.95))
    fig.subplots_adjust(wspace=0.35)
    for ax, (name, v), c in zip(axes, res.items(), [NAVY, TEAL, CORAL]):
        v = np.array(v)
        best = ks[np.argmin(v)] if "lower" in name else ks[np.argmax(v)]
        ax.plot(ks, v, "o-", ms=3, color=c)
        ax.axvline(best, color=GRAY, ls="--", lw=0.8)
        ax.set_title(name.replace(" (", "\n("), fontsize=7.4)
        ax.set_xlabel("number of clusters $k$")
        print(f"  {name}: best k = {best}")
    save(fig, "internal_indices")


# --------------------------------------------------------------------------
# 1.15 Statistical comparison
# --------------------------------------------------------------------------
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}


def benchmark_datasets():
    data = []
    for i, (nf, ni, sep) in enumerate([(10, 4, 0.7), (20, 6, 1.0), (30, 5, 0.6), (15, 10, 1.2),
                                       (25, 8, 0.8), (40, 12, 1.0), (12, 3, 0.9), (50, 15, 0.7)]):
        X, y = make_classification(n_samples=600, n_features=nf, n_informative=ni,
                                   n_redundant=min(4, nf - ni), flip_y=0.05,
                                   class_sep=sep, random_state=100 + i)
        data.append((f"linear-{i + 1}", X, y))
    for i, noise in enumerate([0.25, 0.35]):
        X, y = make_moons(n_samples=600, noise=noise, random_state=200 + i)
        data.append((f"moons-{i + 1}", X, y))
    for i, noise in enumerate([0.08, 0.15]):
        X, y = make_circles(n_samples=600, noise=noise, factor=0.5, random_state=300 + i)
        data.append((f"circles-{i + 1}", X, y))
    return data


def fig_cd_diagram():
    models = {"LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "NB": GaussianNB(),
              "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
              "DT": DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
              "RF": RandomForestClassifier(200, random_state=0)}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    data = benchmark_datasets()
    A = np.array([[cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()
                   for m in models.values()] for _, X, y in data])
    ranks = np.array([rankdata(-row) for row in A])
    avg = ranks.mean(0)
    N, k = A.shape
    chi2, p = friedmanchisquare(*A.T)
    cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    names = list(models)
    with open(os.path.join(OUT, "cd_results.txt"), "w") as fh:
        fh.write("dataset " + " ".join(names) + "\n")
        for (dn, _, _), row in zip(data, A):
            fh.write(dn + " " + " ".join(f"{v:.3f}" for v in row) + "\n")
        fh.write("avg_rank " + " ".join(f"{v:.2f}" for v in avg) + "\n")
        fh.write(f"friedman_chi2 {chi2:.3f} p {p:.2e} CD {cd:.3f}\n")
        ctrl = int(np.argmin(avg))                       # control: the best average rank
        others = [j for j in range(k) if j != ctrl]
        raw = [wilcoxon(A[:, ctrl], A[:, j]).pvalue for j in others]
        order, adj, run = np.argsort(raw), np.empty(len(raw)), 0.0
        for r, j in enumerate(order):                    # Holm's step-down adjustment
            run = max(run, min(1.0, (len(raw) - r) * raw[j])); adj[j] = run
        for j, pr, pa in zip(others, raw, adj):
            fh.write(f"wilcoxon {names[ctrl]}-{names[j]} p {pr:.4f} holm {pa:.4f}\n")
            print(f"  Wilcoxon {names[ctrl]} vs {names[j]}: p={pr:.4f}, Holm-adjusted {pa:.4f}")
    print("  AUC table:")
    for (dn, _, _), row in zip(data, A):
        print("   ", dn, np.round(row, 3))
    print(f"  avg ranks {dict(zip(names, np.round(avg, 2)))}; Friedman chi2={chi2:.2f}, p={p:.2e}; CD={cd:.3f}")
    # draw the critical difference diagram
    order = np.argsort(avg)
    fig, ax = plt.subplots(figsize=(FULL, 1.75))
    ax.set_xlim(0.6, k + 0.4)
    ax.set_ylim(-1.35, 0.6)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color="black", lw=0.8)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.08], color="black", lw=0.8)
        ax.text(r, 0.15, str(r), ha="center", fontsize=7.5)
    ax.plot([1, 1 + cd], [0.45, 0.45], color=CORAL, lw=1.6)
    ax.text(1 + cd / 2, 0.5, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7, color=CORAL)
    half = int(np.ceil(k / 2))
    for pos, i in enumerate(order):
        left = pos < half
        yline = -0.3 - 0.28 * (pos if left else (k - 1 - pos))
        xend = 0.8 if left else k + 0.2
        ax.plot([avg[i], avg[i], xend], [0, yline, yline], color=NAVY, lw=0.8)
        ax.text(xend + (-0.05 if left else 0.05), yline, f"{names[i]} ({avg[i]:.2f})",
                ha="right" if left else "left", va="center", fontsize=7.5)
    # cliques: maximal groups of consecutive methods whose rank difference < CD
    cliques = []
    for a in range(k):
        b = a
        while b + 1 < k and avg[order[b + 1]] - avg[order[a]] < cd:
            b += 1
        if b > a and not any(a >= c0 and b <= c1 for c0, c1 in cliques):
            cliques.append((a, b))
    for j, (a, b) in enumerate(cliques):
        yy = -0.12 - 0.09 * j
        ax.plot([avg[order[a]] - 0.03, avg[order[b]] + 0.03], [yy, yy], color=CORAL, lw=2.4)
    save(fig, "cd_diagram")


ALL = [fig_polyfits, fig_bias_variance, fig_knn, fig_learning_curves, fig_prevalence,
       fig_signal_detection, fig_roc_space, fig_roc_hull, fig_roc_construction,
       fig_roc_models, fig_gains_lift, fig_reliability, fig_calibration_methods,
       fig_pr_vs_roc, fig_smote, fig_cost_threshold, fig_cost_curves,
       fig_regression_diagnostics, fig_regression_losses, fig_conformal, fig_open_set,
       fig_detection_pr, fig_dice_iou, fig_silhouette, fig_internal_indices, fig_cd_diagram]

if __name__ == "__main__":
    use_style()
    only = set(sys.argv[2:])
    for f in ALL:
        if not only or f.__name__ in only:
            print(f.__name__)
            f()
