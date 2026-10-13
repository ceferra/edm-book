"""Generate all the figures of Unit 2 (Hybrid models).

Usage:
    python make_figures.py                     # writes ../../figures/ch02/*.pdf
    python make_figures.py OUT_DIR [fig_name ...]

All randomness is seeded, so figures and quoted numbers are reproducible.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom, rankdata, friedmanchisquare
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ch01"))
from edm_style import (use_style, NAVY, BRONZE, TEAL, CORAL, GOLD, INDIGO, GRAY, LIGHT,  # noqa: E402
                       CYCLE, FULL, HALF)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "figures", "ch02")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"[saved] {name}.pdf")


def member_proba(ens, X):
    """Positive-class probabilities of every member of a bagging-type ensemble."""
    feats = getattr(ens, "estimators_features_", None)
    if feats is None:
        feats = [slice(None)] * len(ens.estimators_)
    return np.array([e.predict_proba(X[:, f])[:, 1] for e, f in zip(ens.estimators_, feats)])


# --------------------------------------------------------------------------
def fig_condorcet():
    T = np.arange(1, 102, 2)
    fig, ax = plt.subplots(figsize=(HALF + 1.0, 2.5))
    for p, c in [(0.55, GOLD), (0.6, CORAL), (0.7, TEAL)]:
        ax.plot(T, binom.sf(T // 2, T, p), color=c, label=f"independent voters, p = {p}")
    rho, p = 0.3, 0.6
    ax.plot(T, rho * p + (1 - rho) * binom.sf(T // 2, T, p), "--", color=CORAL,
            label=f"correlated voters ($\\rho$ = {rho}), p = {p}")
    ax.plot(T, binom.sf(T // 2, T, 0.45), ":", color=GRAY, label="independent voters, p = 0.45")
    ax.set_xlabel("number of voters $T$")
    ax.set_ylabel("P(majority is correct)")
    ax.set_ylim(0.1, 1.01)
    ax.legend(loc="lower right", fontsize=6.3)
    for T0 in (1, 11, 51, 101):
        print(f"  T={T0}: p=0.6 -> {binom.sf(T0 // 2, T0, 0.6):.3f}; correlated -> "
              f"{rho * p + (1 - rho) * binom.sf(T0 // 2, T0, p):.3f}")
    save(fig, "condorcet")


def fig_boundaries():
    X, y = make_moons(n_samples=400, noise=0.30, random_state=0)
    Xt, yt = make_moons(n_samples=4000, noise=0.30, random_state=1)
    models = [("one decision stump", DecisionTreeClassifier(max_depth=1)),
              ("AdaBoost, 100 stumps", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                          n_estimators=100, random_state=0)),
              ("one unpruned tree", DecisionTreeClassifier(random_state=0)),
              ("bagging, 100 trees", BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,
                                                       random_state=0)),
              ("random forest, 200 trees", RandomForestClassifier(200, random_state=0)),
              ("gradient boosting", HistGradientBoostingClassifier(random_state=0))]
    xx, yy = np.meshgrid(np.linspace(-1.7, 2.7, 300), np.linspace(-1.3, 1.8, 220))
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.38)
    for ax, (name, m) in zip(axes.ravel(), models):
        m.fit(X, y)
        Z = m.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=["#DCE3E7", "#F7DDD6"])
        ax.contour(xx, yy, Z, levels=[0.5], colors=[NAVY], linewidths=0.9)
        ax.scatter(*X[y == 0].T, s=4, color=NAVY)
        ax.scatter(*X[y == 1].T, s=4, color=CORAL)
        acc = m.score(Xt, yt)
        ax.set_title(f"{name}\ntest accuracy {acc:.3f}", fontsize=7.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        print(f"  {name}: {acc:.3f}")
    save(fig, "boundaries")


def fig_ensemble_size():
    X, y = make_classification(n_samples=3000, n_features=20, n_informative=10, n_redundant=4,
                               flip_y=0.05, class_sep=1.0, random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    Tmax = 300
    T = np.arange(1, Tmax + 1)
    tree_err = 1 - DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr).score(X_te, y_te)
    curves = {}
    for name, ens in [("bagging", BaggingClassifier(DecisionTreeClassifier(), n_estimators=Tmax,
                                                    random_state=0)),
                      ("random forest", RandomForestClassifier(Tmax, random_state=0)),
                      ("extra trees", ExtraTreesClassifier(Tmax, random_state=0))]:
        P = np.cumsum(member_proba(ens.fit(X_tr, y_tr), X_te), axis=0) / T[:, None]
        curves[name] = np.mean((P > 0.5) != y_te, axis=1)
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=Tmax,
                             random_state=0).fit(X_tr, y_tr)
    curves["AdaBoost (depth-2 trees)"] = np.array([np.mean(p != y_te) for p in ada.staged_predict(X_te)])
    gb = GradientBoostingClassifier(n_estimators=Tmax, random_state=0).fit(X_tr, y_tr)
    curves["gradient boosting"] = np.array([np.mean(p != y_te) for p in gb.staged_predict(X_te)])
    fig, ax = plt.subplots(figsize=(HALF + 1.1, 2.6))
    for (name, err), c in zip(curves.items(), [GOLD, TEAL, INDIGO, CORAL, NAVY]):
        ax.plot(np.arange(1, len(err) + 1), err, color=c, lw=1.3, label=name)
        print(f"  {name}: error at T=1 {err[0]:.3f}, T=10 {err[9]:.3f}, T=100 {err[99]:.3f}, "
              f"T={len(err)} {err[-1]:.3f}")
    ax.axhline(tree_err, color=GRAY, ls=":", lw=1, label="single unpruned tree")
    print(f"  single tree error {tree_err:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("number of members $T$")
    ax.set_ylabel("test error")
    ax.set_ylim(0, 0.35)
    ax.legend(loc="upper right", fontsize=6.4)
    save(fig, "ensemble_size")


def fig_kappa_error():
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, n_redundant=4,
                               flip_y=0.05, class_sep=1.0, random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    ens = {"bagging": (BaggingClassifier(DecisionTreeClassifier(), n_estimators=40, random_state=0), GOLD),
           "random forest": (RandomForestClassifier(40, random_state=0), TEAL),
           "AdaBoost": (AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=40,
                                           random_state=0), CORAL)}
    fig, ax = plt.subplots(figsize=(HALF + 0.9, 2.7))
    for name, (m, c) in ens.items():
        m.fit(X_tr, y_tr)
        if name == "AdaBoost":
            pred = np.array([e.predict(X_te) for e in m.estimators_])
        else:
            pred = (member_proba(m, X_te) > 0.5).astype(int)
        err = np.mean(pred != y_te, axis=1)
        ks, es = [], []
        for i in range(len(pred)):
            for j in range(i + 1, len(pred)):
                po = np.mean(pred[i] == pred[j])
                pi, pj = pred[i].mean(), pred[j].mean()
                pe = pi * pj + (1 - pi) * (1 - pj)
                ks.append((po - pe) / (1 - pe))
                es.append((err[i] + err[j]) / 2)
        ax.scatter(ks, es, s=4, color=c, alpha=0.45, label=f"{name} (ensemble error {1 - m.score(X_te, y_te):.3f})")
        print(f"  {name}: mean kappa {np.mean(ks):.3f}, mean member error {err.mean():.3f}, "
              f"ensemble error {1 - m.score(X_te, y_te):.3f}")
    ax.set_xlabel(r"pairwise agreement $\kappa$ (lower = more diverse)")
    ax.set_ylabel("mean error of the pair")
    leg = ax.legend(loc="upper left", fontsize=6.3, markerscale=2.5)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    save(fig, "kappa_error")


def fig_noise():
    levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    models = {"single tree": lambda: DecisionTreeClassifier(random_state=0),
              "bagging": lambda: BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,
                                                   n_jobs=-1, random_state=0),
              "random forest": lambda: RandomForestClassifier(200, n_jobs=-1, random_state=0),
              "AdaBoost (depth-3 trees)": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                                                     n_estimators=200, random_state=0)}
    acc = {k: np.zeros(len(levels)) for k in models}
    seeds = range(10, 14)
    for s in seeds:
        X, y = make_classification(n_samples=1500, n_features=20, n_informative=8, n_redundant=4,
                                   flip_y=0.0, class_sep=1.0, random_state=s)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=1 / 3, random_state=0, stratify=y)
        rng = np.random.default_rng(s)
        for li, lev in enumerate(levels):
            yn = y_tr.copy()
            flip = rng.choice(len(yn), int(lev * len(yn)), replace=False)
            yn[flip] = 1 - yn[flip]
            for name, mk in models.items():
                acc[name][li] += mk().fit(X_tr, yn).score(X_te, y_te) / len(seeds)
    fig, ax = plt.subplots(figsize=(HALF + 0.9, 2.5))
    for (name, a), c in zip(acc.items(), [GRAY, GOLD, TEAL, CORAL]):
        ax.plot(np.array(levels) * 100, a, "o-", ms=3, color=c, label=name)
        print(f"  {name}: " + " ".join(f"{v:.3f}" for v in a))
    ax.set_xlabel("label noise in the training set (%)")
    ax.set_ylabel("test accuracy (clean labels)")
    ax.legend(loc="lower left", fontsize=6.5)
    save(fig, "noise")


def cautious_predict(P, K, w):
    tau = (1 - np.asarray(K)) * w + np.asarray(K)
    ok = P >= tau
    pred = np.where(ok, P / tau, -np.inf).argmax(axis=1)
    pred[~ok.any(axis=1)] = -1
    return pred


def fig_cautious():
    # (a) accuracy response of three classifiers on a balanced problem
    X, y = make_classification(n_samples=4000, n_features=20, n_informative=6, n_redundant=4,
                               flip_y=0.05, class_sep=0.8, random_state=5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    models = {"logistic regression": (LogisticRegression(max_iter=1000), NAVY),
              "naive Bayes": (GaussianNB(), GOLD),
              "decision tree (depth 6)": (DecisionTreeClassifier(max_depth=6, random_state=0), CORAL)}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.65))
    fig.subplots_adjust(wspace=0.3)
    grid = np.linspace(0, 0.9, 46)
    for name, (m, c) in models.items():
        P = m.fit(X_tr, y_tr).predict_proba(X_te)
        conf, correct = P.max(axis=1), P.argmax(axis=1) == y_te
        cs = correct[np.argsort(-conf, kind="stable")]
        risk = 1 - np.cumsum(cs) / np.arange(1, len(cs) + 1)
        accs = [cs[: max(1, int(round((1 - a) * len(cs))))].mean() for a in grid]
        axes[0].plot(grid * 100, accs, color=c, label=f"{name} (AURC {risk.mean():.3f})")
        print(f"  {name}: acc 0% {accs[0]:.3f}, 20% {accs[10]:.3f}, 50% {accs[25]:.3f}, AURC {risk.mean():.3f}")
    axes[0].set_xlabel("abstention (%)")
    axes[0].set_ylabel("accuracy on the covered examples")
    axes[0].set_title("(a) Accuracy response of three classifiers", fontsize=8)
    axes[0].set_xlim(0, 90)
    axes[0].set_ylim(0.7, 1.005)
    axes[0].legend(loc="lower right", fontsize=6.2)
    # (b) unequal costs: uniform versus cost-matched class bias
    X, y = make_classification(n_samples=6000, n_features=20, n_informative=6, n_redundant=4,
                               weights=[0.8], flip_y=0.03, class_sep=0.8, random_state=6)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    P = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)
    c_fp, c_fn, c_r = 1.0, 4.0, 0.5
    t = c_fp / (c_fp + c_fn)

    def cost_of(pred):
        return (c_fp * np.sum((pred == 1) & (y_te == 0)) + c_fn * np.sum((pred == 0) & (y_te == 1))
                + c_r * np.sum(pred == -1)) / len(y_te)

    for K, c, lab in [((0.5, 0.5), NAVY, "uniform $K=(0.5,0.5)$"),
                      ((1 - t, t), CORAL, f"cost-matched $K=({1 - t:.1f},{t:.1f})$")]:
        pts = np.array([(np.mean(pr == -1), cost_of(pr))
                        for pr in (cautious_predict(P, K, w) for w in np.linspace(0, 0.98, 99))])
        axes[1].plot(pts[:, 0] * 100, pts[:, 1], color=c, lw=1.4, label=lab)
        print(f"  K={K}: cost at w=0 {pts[0, 1]:.3f}; min cost {pts[:, 1].min():.3f} at abstention {pts[np.argmin(pts[:, 1]), 0]:.3f}")
    w_star = 1 - c_r * (c_fp + c_fn) / (c_fp * c_fn)
    pr = cautious_predict(P, (1 - t, t), w_star)
    bayes = np.array([0, 1, -1])[np.c_[P[:, 1] * c_fn, P[:, 0] * c_fp, np.full(len(P), c_r)].argmin(axis=1)]
    axes[1].plot(np.mean(pr == -1) * 100, cost_of(pr), "*", ms=10, color=GOLD, mec=NAVY, mew=0.6,
                 label=f"optimal reject rule ($w={w_star:.3f}$)", zorder=5)
    print(f"  optimal: w*={w_star:.3f}, abstention {np.mean(pr == -1):.3f}, cost {cost_of(pr):.3f}; "
          f"same decisions as Bayes rule: {np.mean(pr == bayes):.4f}")
    axes[1].set_xlabel("abstention (%)")
    axes[1].set_ylabel("average cost per example")
    axes[1].set_title("(b) Costs $c_{FP}=1$, $c_{FN}=4$, abstention $0.5$", fontsize=8)
    axes[1].set_xlim(0, 90)
    axes[1].legend(loc="upper right", fontsize=6.2)
    save(fig, "cautious")


def fig_delegation():
    X, y = make_classification(n_samples=6000, n_features=10, n_informative=6, n_redundant=2,
                               flip_y=0.03, class_sep=1.0, random_state=8)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    first = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
    forest = RandomForestClassifier(200, random_state=0).fit(X_tr, y_tr)
    oof = lambda m, A, b: cross_val_predict(m, A, b, cv=5, method="predict_proba").max(axis=1)
    c_tr = oof(DecisionTreeClassifier(max_depth=4, random_state=0), X_tr, y_tr)   # out of fold
    c_te = first.predict_proba(X_te).max(axis=1)
    p1, pf = first.predict(X_te), forest.predict(X_te)
    res = {"cascade": [], "specialist": [], "rebound": []}
    for th in np.r_[-np.inf, np.unique(c_tr)]:
        hard_tr, hard_te = c_tr <= th, c_te <= th
        frac = hard_te.mean()
        res["cascade"].append((frac, np.mean(np.where(hard_te, pf, p1) == y_te)))
        if hard_tr.sum() >= 50 and len(np.unique(y_tr[hard_tr])) == 2:
            spec = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr[hard_tr], y_tr[hard_tr])
            ps = spec.predict(X_te)
            res["specialist"].append((frac, np.mean(np.where(hard_te, ps, p1) == y_te)))
            t2 = np.quantile(oof(DecisionTreeClassifier(max_depth=4, random_state=0),
                                 X_tr[hard_tr], y_tr[hard_tr]), 0.3)
            back = spec.predict_proba(X_te).max(axis=1) <= t2
            res["rebound"].append((frac, np.mean(np.where(hard_te & ~back, ps, p1) == y_te)))
    fig, ax = plt.subplots(figsize=(HALF + 1.2, 2.9))
    for key, c, mk, lab in [("cascade", TEAL, "o", "delegate to a random forest (cascade)"),
                            ("specialist", CORAL, "s", "delegate to a specialist tree"),
                            ("rebound", GOLD, "^", "specialist tree + round rebound")]:
        pts = np.array(sorted(set(res[key])))
        ax.plot(pts[:, 0] * 100, pts[:, 1], marker=mk, ms=3.5, color=c, lw=1.3, label=lab)
        near = pts[np.argmin(np.abs(pts[:, 0] - 0.3))]
        print(f"  {key}: at {near[0]:.3f} delegated -> acc {near[1]:.3f}; best {pts[:, 1].max():.3f} at {pts[np.argmax(pts[:, 1]), 0]:.3f}")
    ax.axhline(first.score(X_te, y_te), color=GRAY, ls=":", lw=1.1, label="small tree alone")
    ax.axhline(forest.score(X_te, y_te), color=NAVY, ls="--", lw=1.1, label="random forest alone")
    print(f"  tree alone {first.score(X_te, y_te):.3f}; forest alone {forest.score(X_te, y_te):.3f}")
    ax.set_xlabel("test examples delegated to the second level (%)")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.1)
    save(fig, "delegation")


def fig_mimetic():
    X, y = make_classification(n_samples=4600, n_features=8, n_informative=5, n_redundant=2,
                               flip_y=0.03, class_sep=1.0, random_state=7)
    X_tr, y_tr, X_te, y_te = X[:600], y[:600], X[600:], y[600:]
    oracle = RandomForestClassifier(300, random_state=0).fit(X_tr, y_tr)
    ref = oracle.predict(X_te)
    direct = DecisionTreeClassifier(max_leaf_nodes=32, random_state=0).fit(X_tr, y_tr)
    sizes = [600, 1000, 2000, 5000, 10000, 20000, 50000]
    acc, fid = [], []
    rng = np.random.default_rng(0)
    for n in sizes:
        Xs = X_tr[rng.integers(0, len(X_tr), n)]
        Xs = Xs + rng.normal(0, 0.3, Xs.shape) * X_tr.std(axis=0)
        t = DecisionTreeClassifier(max_leaf_nodes=32, random_state=0).fit(Xs, oracle.predict(Xs))
        acc.append(t.score(X_te, y_te))
        fid.append(np.mean(t.predict(X_te) == ref))
    fig, ax = plt.subplots(figsize=(HALF + 0.9, 2.5))
    ax.plot(sizes, acc, "o-", ms=3, color=CORAL, label="mimetic tree: accuracy")
    ax.plot(sizes, fid, "s--", ms=3, color=TEAL, label="mimetic tree: fidelity to the forest")
    ax.axhline(oracle.score(X_te, y_te), color=NAVY, ls=":", lw=1.2, label="random forest (oracle): accuracy")
    ax.axhline(direct.score(X_te, y_te), color=GRAY, ls="-.", lw=1.2, label="tree on the original data: accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("number of synthetic examples labelled by the oracle")
    ax.set_ylabel("proportion")
    ax.set_ylim(0.80, 0.95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.2)
    print(f"  oracle acc {oracle.score(X_te, y_te):.3f}; direct tree {direct.score(X_te, y_te):.3f}; "
          f"mimetic acc {np.round(acc, 3)}; fidelity {np.round(fid, 3)}")
    save(fig, "mimetic")


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


def fig_cd_ensembles():
    base = [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
            ("nb", GaussianNB()),
            ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(15))),
            ("dt", DecisionTreeClassifier(min_samples_leaf=5, random_state=0))]
    models = {"Tree": DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
              "Bagging": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0),
              "RF": RandomForestClassifier(100, random_state=0),
              "ExtraTrees": ExtraTreesClassifier(100, random_state=0),
              "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100,
                                             random_state=0),
              "GBoost": HistGradientBoostingClassifier(random_state=0),
              "Stacking": StackingClassifier(base, final_estimator=LogisticRegression(), cv=5)}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    data = benchmark_datasets()
    A = np.array([[cross_val_score(m, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
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
        fh.write("mean_auc " + " ".join(f"{v:.3f}" for v in A.mean(0)) + "\n")
        fh.write("avg_rank " + " ".join(f"{v:.2f}" for v in avg) + "\n")
        fh.write(f"friedman_chi2 {chi2:.3f} p {p:.2e} CD {cd:.3f}\n")
    print("  mean AUC", dict(zip(names, np.round(A.mean(0), 3))))
    print(f"  avg ranks {dict(zip(names, np.round(avg, 2)))}; chi2={chi2:.2f}, p={p:.2e}; CD={cd:.3f}")
    order = np.argsort(avg)
    fig, ax = plt.subplots(figsize=(FULL, 1.95))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.55, 0.6)
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
        yline = -0.3 - 0.25 * (pos if left else (k - 1 - pos))
        xend = 0.7 if left else k + 0.3
        ax.plot([avg[i], avg[i], xend], [0, yline, yline], color=NAVY, lw=0.8)
        ax.text(xend + (-0.05 if left else 0.05), yline, f"{names[i]} ({avg[i]:.2f})",
                ha="right" if left else "left", va="center", fontsize=7.3)
    cliques = []
    for a in range(k):
        b = a
        while b + 1 < k and avg[order[b + 1]] - avg[order[a]] < cd:
            b += 1
        if b > a and not any(a >= c0 and b <= c1 for c0, c1 in cliques):
            cliques.append((a, b))
    for j, (a, b) in enumerate(cliques):
        yy = -0.12 - 0.08 * j
        ax.plot([avg[order[a]] - 0.03, avg[order[b]] + 0.03], [yy, yy], color=CORAL, lw=2.4)
    save(fig, "cd_ensembles")


ALL = [fig_condorcet, fig_boundaries, fig_ensemble_size, fig_kappa_error, fig_noise,
       fig_cautious, fig_delegation, fig_mimetic, fig_cd_ensembles]

if __name__ == "__main__":
    use_style()
    only = set(sys.argv[2:])
    for f in ALL:
        if not only or f.__name__ in only:
            print(f.__name__)
            f()
