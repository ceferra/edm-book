"""Generate all the figures of Unit 3 (Reliable models).

Usage:
    python make_figures.py                     # writes ../../figures/ch03/*.pdf
    python make_figures.py OUT_DIR [fig_name ...]
All randomness is seeded, so the figures and the numbers quoted in the book are reproducible.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_diabetes, load_digits, make_friedman1
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ch01"))
sys.path.insert(0, os.path.join(HERE, "listings"))
from edm_style import (use_style, NAVY, BRONZE, TEAL, CORAL, GOLD, INDIGO, GRAY,  # noqa: E402
                       LIGHT, CYCLE, FULL, HALF)
from credit import credit_data, four_policies  # noqa: E402
from apartments import apartments  # noqa: E402
from loans import loan_data  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "figures", "ch03")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"[saved] {name}.pdf")


# --------------------------------------------------------------------------- fairness
def fig_fairness_credit():
    X, A, Y = credit_data(40000)
    X_tr, X_te, A_tr, A_te, Y_tr, Y_te = train_test_split(X, A, Y, test_size=0.5, random_state=0)
    X_va, X_te, A_va, A_te, Y_va, Y_te = train_test_split(X_te, A_te, Y_te, test_size=0.5,
                                                          random_state=1)
    model = LogisticRegression().fit(X_tr, Y_tr)
    R = model.predict_proba(X_te)[:, 1]
    pol = four_policies(model.predict_proba(X_va)[:, 1], A_va, Y_va)   # chosen on validation
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3))
    fig.subplots_adjust(wspace=0.38)
    names = ["group a (70%)", "group b (30%)"]
    for g, c in [(0, NAVY), (1, CORAL)]:
        axes[0].hist(R[A_te == g], bins=30, range=(0, 1), density=True, alpha=0.55, color=c, label=names[g])
        fpr, tpr, _ = roc_curve(Y_te[A_te == g], R[A_te == g])
        axes[1].plot(fpr, tpr, color=c, lw=1.2, label=names[g])
        frac, mean_p = calibration_curve(Y_te[A_te == g], R[A_te == g], n_bins=10)
        axes[2].plot(mean_p, frac, "o-", ms=2.5, color=c, label=names[g])
    markers = {"max profit": "o", "single threshold": "s", "independence": "^", "equal opportunity": "D"}
    for name, t in pol.items():
        for g, c in [(0, NAVY), (1, CORAL)]:
            m = A_te == g
            acc = R[m] >= t[g]
            axes[1].plot(acc[Y_te[m] == 0].mean(), acc[Y_te[m] == 1].mean(), markers[name], color=c,
                         ms=4.5, mfc="white", mew=1.1, label=name if g == 0 else None)
    axes[0].set_xlabel("score $R$")
    axes[0].set_yticks([])
    axes[0].set_title("(a) Score distributions", fontsize=7.8)
    axes[0].legend(loc="upper left", fontsize=6.0)
    axes[1].plot([0, 1], [0, 1], ":", color=GRAY, lw=0.8)
    axes[1].set_xlabel("false positive rate")
    axes[1].set_ylabel("true positive rate")
    axes[1].set_title("(b) ROC by group and policies", fontsize=7.8)
    h, l = axes[1].get_legend_handles_labels()
    axes[1].legend(h[2:], l[2:], loc="lower right", fontsize=5.6)
    axes[2].plot([0, 1], [0, 1], ":", color=GRAY, lw=0.8)
    axes[2].set_xlabel("mean score")
    axes[2].set_ylabel("observed repayment rate")
    axes[2].set_title("(c) Calibration by group", fontsize=7.8)
    axes[2].legend(loc="upper left", fontsize=6.0)
    for name, t in pol.items():
        print(f"  {name}: thresholds {t[0]:.3f}, {t[1]:.3f}")
    save(fig, "fairness_credit")


# --------------------------------------------------------------------------- interpretable models
def fig_linear():
    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    Xs = (X - X.mean()) / X.std()
    D = np.c_[np.ones(len(Xs)), Xs.values]
    beta = np.linalg.lstsq(D, y.values, rcond=None)[0]
    res = y.values - D @ beta
    se = np.sqrt(np.diag(res @ res / (D.shape[0] - D.shape[1]) * np.linalg.inv(D.T @ D)))
    order = np.argsort(beta[1:])
    names = np.array(X.columns)[order]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.7))
    fig.subplots_adjust(wspace=0.3)
    ax = axes[0]
    ax.errorbar(beta[1:][order], np.arange(10), xerr=1.96 * se[1:][order], fmt="o", ms=3.5,
                color=NAVY, ecolor=TEAL, capsize=2)
    ax.axvline(0, color=GRAY, ls=":", lw=1)
    ax.set_yticks(np.arange(10))
    ax.set_yticklabels(names)
    ax.set_xlabel("weight (per standard deviation) with 95% CI")
    ax.set_title("(a) Weight plot", fontsize=8)
    effects = Xs.values * beta[1:]
    ax = axes[1]
    ax.boxplot([effects[:, j] for j in order], vert=False, widths=0.6, flierprops=dict(ms=1.5),
               medianprops=dict(color=CORAL))
    i = int(np.argmax(D @ beta))
    ax.plot(effects[i, order], np.arange(1, 11), "x", color=CORAL, ms=6, mew=1.5, label=f"instance {i}")
    ax.set_yticks(np.arange(1, 11))
    ax.set_yticklabels(names)
    ax.axvline(0, color=GRAY, ls=":", lw=1)
    ax.set_xlabel("effect = weight $\\times$ feature value")
    ax.set_title("(b) Effect plot", fontsize=8)
    ax.legend(loc="lower right", fontsize=6.5)
    print(f"  instance {i}: prediction {D[i] @ beta:.1f}, mean {y.mean():.1f}")
    save(fig, "linear_weights")


def friedman_model():
    X, y = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    gbr = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                    random_state=0).fit(X_tr, y_tr)
    return gbr, X_tr, X_te, y_tr, y_te


def partial_dependence(model, X, j, grid):
    out = []
    for v in grid:
        Xv = X.copy()
        Xv[:, j] = v
        out.append(model.predict(Xv).mean())
    return np.array(out)


def pd_values(model, X, cols):
    out = np.empty(len(X))
    for i in range(len(X)):
        Xi = X.copy()
        Xi[:, cols] = X[i, cols]
        out[i] = model.predict(Xi).mean()
    return out - out.mean()


def h2(model, X, j, k):
    pj, pk, pjk = pd_values(model, X, [j]), pd_values(model, X, [k]), pd_values(model, X, [j, k])
    return np.sum((pjk - pj - pk) ** 2) / np.sum(pjk ** 2)


def fig_pdp_ice():
    gbr, X_tr, X_te, _, _ = friedman_model()
    Xs = X_te[:400]
    grid = np.linspace(0, 1, 31)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    fig.subplots_adjust(wspace=0.28)
    ax = axes[0]
    for j, truth, c, lab in [(2, lambda v: 20 * (v - 0.5) ** 2, TEAL, "$x_3$"),
                             (3, lambda v: 10 * v, CORAL, "$x_4$"),
                             (5, lambda v: 0 * v, GRAY, "$x_6$ (noise)")]:
        pdv = partial_dependence(gbr, Xs, j, grid)
        tv = truth(grid)
        ax.plot(grid, pdv - pdv.mean(), color=c, lw=1.6, label=f"PDP of {lab}")
        ax.plot(grid, tv - tv.mean(), "--", color=c, lw=0.9)
    ax.set_xlabel("feature value")
    ax.set_ylabel("centred partial dependence")
    ax.set_title("(a) PDPs (dashed: true effect)", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.3)
    ax = axes[1]
    idx = np.arange(80)
    cmap = plt.get_cmap("viridis")
    for i in idx:
        Xi = np.repeat(Xs[i:i + 1], len(grid), axis=0)
        Xi[:, 0] = grid
        p = gbr.predict(Xi)
        ax.plot(grid, p - p[0], color=cmap(Xs[i, 1]), lw=0.6, alpha=0.8)
    pdv = partial_dependence(gbr, Xs, 0, grid)
    ax.plot(grid, pdv - pdv[0], color="black", lw=2, label="PDP (centred)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("value of $x_2$", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("change in prediction")
    ax.set_title("(b) Centred ICE curves of $x_1$", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.3)
    save(fig, "pdp_ice")


def fig_importance_h():
    X, y = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestRegressor(300, random_state=0).fit(X_tr, y_tr)
    pi = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0)
    imp = rf.feature_importances_
    perm = np.clip(pi.importances_mean, 0, None)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    fig.subplots_adjust(wspace=0.3)
    ax = axes[0]
    k = np.arange(10)
    ax.bar(k - 0.2, imp / imp.sum(), 0.4, color=GOLD, label="impurity (training data)")
    ax.bar(k + 0.2, perm / perm.sum(), 0.4, color=NAVY, label="permutation (test data)")
    ax.set_xticks(k)
    ax.set_xticklabels([f"$x_{{{i + 1}}}$" for i in k])
    ax.set_ylabel("normalised importance")
    ax.set_title("(a) Feature importance of a random forest", fontsize=8)
    ax.legend(loc="upper right", fontsize=6.3)
    gbr, _, X_te2, _, _ = friedman_model()
    Xs = X_te2[:120]
    pairs = [(0, 1), (0, 2), (2, 3), (3, 4), (0, 5)]
    vals = [h2(gbr, Xs, j, kk) for j, kk in pairs]
    ax = axes[1]
    ax.bar(range(len(pairs)), vals, color=[CORAL if v > 0.05 else TEAL for v in vals])
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"$x_{j + 1}$,$x_{kk + 1}$" for j, kk in pairs])
    ax.set_ylabel("$H^2$")
    ax.set_title("(b) Two-way interaction strength", fontsize=8)
    print("  impurity", np.round(imp / imp.sum(), 3))
    print("  permutation", np.round(pi.importances_mean, 3))
    print("  H2", dict(zip([f"x{j + 1}x{kk + 1}" for j, kk in pairs], np.round(vals, 3))))
    save(fig, "importance_h")


# --------------------------------------------------------------------------- local explanations
def shapley_exact(f, x, background):
    from itertools import combinations
    from math import factorial
    p = len(x)

    def v(S):
        Z = background.copy()
        Z[:, list(S)] = x[list(S)]
        return f(Z).mean()
    phi = np.zeros(p)
    for j in range(p):
        others = [k for k in range(p) if k != j]
        for size in range(p):
            for S in combinations(others, size):
                w = factorial(len(S)) * factorial(p - len(S) - 1) / factorial(p)
                phi[j] += w * (v(S + (j,)) - v(S))
    return phi


def lime_explain(f, X_train, x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    Z = rng.normal(mu, sd, (n, len(x)))
    d = np.linalg.norm((Z - x) / sd, axis=1)
    w = np.exp(-d ** 2 / (0.75 * np.sqrt(len(x))) ** 2)
    g = Ridge(alpha=1.0).fit((Z - mu) / sd, f(Z), sample_weight=w)
    return g.coef_


def fig_lime_shap():
    X, y = apartments()
    model = GradientBoostingRegressor(random_state=0).fit(X, y)
    x = np.array([1.0, 0.0, 50.0, 2.0])
    bg = X[np.random.default_rng(1).choice(len(X), 300, replace=False)]
    phi = shapley_exact(model.predict, x, bg)
    base, pred = model.predict(bg).mean(), model.predict(x[None])[0]
    labels = ["park nearby", "cat forbidden", "area 50 m$^2$", "2nd floor"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    fig.subplots_adjust(wspace=0.55)
    ax = axes[0]
    ax.barh(range(4), phi, color=[TEAL if v > 0 else CORAL for v in phi])
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels)
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("contribution (thousand EUR)")
    ax.set_title(f"(a) Shapley values: {pred:.0f} = {base:.0f} + ({phi.sum():+.0f})", fontsize=7.8)
    coef = lime_explain(model.predict, X, x)
    ax = axes[1]
    ax.barh(range(4), coef, color=[TEAL if v > 0 else CORAL for v in coef])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["park", "cat allowed", "area", "floor"])
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("local weight (thousand EUR per SD)")
    ax.set_title("(b) LIME weights around the same flat", fontsize=7.8)
    print(f"  base {base:.1f} pred {pred:.1f} phi {np.round(phi, 1)} sum {phi.sum():.1f}; lime {np.round(coef, 1)}")
    save(fig, "lime_shap")


def fig_counterfactual():
    import l15_counterfactual_explanation as cfx        # same model, applicant and search
    f, peter = cfx.f, cfx.peter
    cf = cfx.counterfactual(peter, target=0.51, tol=0.005)
    ii, dd = np.meshgrid(np.linspace(10, 50, 200), np.linspace(0, 0.8, 200))
    G = np.c_[ii.ravel(), dd.ravel(), np.full(ii.size, peter[2]), np.full(ii.size, peter[3])]
    P = f(G).reshape(ii.shape)
    fig, ax = plt.subplots(figsize=(HALF + 0.6, 2.6))
    cs = ax.contourf(ii, dd, P, levels=np.linspace(0, 1, 11), cmap="RdYlGn", alpha=0.55)
    ax.contour(ii, dd, P, levels=[0.5], colors=[NAVY], linewidths=1.2)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("P(repay)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.plot(*peter[:2], "o", color="black", ms=5)
    ax.annotate(f"Peter: P = {f(peter[None])[0]:.2f}", peter[:2], xytext=(33, 0.68), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.plot(*cf[:2], "*", color=GOLD, mec="black", mew=0.5, ms=11)
    ax.annotate("", xy=cf[:2], xytext=peter[:2], arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.annotate(f"counterfactual: P = {f(cf[None])[0]:.2f}", cf[:2], xytext=(36, 0.05), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("income (thousand EUR)")
    ax.set_ylabel("debt-to-income ratio")
    print(f"  peter P={f(peter[None])[0]:.3f}; cf={np.round(cf, 3)} P={f(cf[None])[0]:.3f}")
    save(fig, "counterfactual")


# --------------------------------------------------------------------------- example-based
def rbf(A, B, gamma):
    return np.exp(-gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def mmd_critic(X, m, c, gamma=1.0):
    K = rbf(X, X, gamma)
    colmean = K.mean(axis=0)
    protos = []
    for _ in range(m):
        best, best_val = None, np.inf
        for i in range(len(X)):
            if i in protos:
                continue
            S = protos + [i]
            val = K[np.ix_(S, S)].mean() - 2 * colmean[S].mean()
            if val < best_val:
                best, best_val = i, val
        protos.append(best)
    witness = colmean - K[:, protos].mean(axis=1)
    crit = [i for i in np.argsort(-np.abs(witness)) if i not in protos][:c]
    return protos, crit, witness


def fig_mmd():
    from l16_mmd_critic import X, protos, crit           # same data and selection as the listing
    fig, ax = plt.subplots(figsize=(HALF + 0.4, 2.5))
    ax.scatter(*X.T, s=5, color=LIGHT, edgecolor=GRAY, lw=0.2, label="data")
    ax.scatter(*X[protos].T, s=60, color=TEAL, edgecolor="black", lw=0.6, label="prototypes", zorder=3)
    ax.scatter(*X[crit].T, s=60, marker="^", color=CORAL, edgecolor="black", lw=0.6, label="criticisms", zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="upper left", fontsize=6.5)
    save(fig, "mmd_critic")


def fig_influence():
    rng = np.random.default_rng(7)
    x = np.r_[rng.uniform(0, 5, 30), 12.0]
    y = np.r_[2 * x[:30] + 1 + rng.normal(0, 1.2, 30), 6.0]
    X = x[:, None]
    full = LinearRegression().fit(X, y)
    yhat = full.predict(X)
    s2 = np.sum((y - yhat) ** 2) / (len(y) - 2)
    cook = np.array([np.sum((yhat - LinearRegression().fit(np.delete(X, i, 0), np.delete(y, i)).predict(X)) ** 2)
                     / (2 * s2) for i in range(len(y))])
    wo = LinearRegression().fit(X[:30], y[:30])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.35))
    ax = axes[0]
    ax.scatter(x[:30], y[:30], s=10, color=NAVY)
    ax.scatter([12], [6], s=40, color=CORAL, marker="D", label="influential instance")
    xx = np.linspace(0, 12.5, 2)
    ax.plot(xx, full.predict(xx[:, None]), color=CORAL, label=f"all data (slope {full.coef_[0]:.2f})")
    ax.plot(xx, wo.predict(xx[:, None]), "--", color=TEAL, label=f"without it (slope {wo.coef_[0]:.2f})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper left", fontsize=6.3)
    ax.set_title("(a) One instance changes the model", fontsize=8)
    ax = axes[1]
    ax.stem(np.arange(len(y)), cook, linefmt=GRAY, markerfmt="o", basefmt=" ")
    ax.plot(30, cook[30], "D", color=CORAL)
    ax.set_xlabel("training instance")
    ax.set_ylabel("Cook's distance")
    ax.set_title("(b) Deletion diagnostics", fontsize=8)
    print(f"  slopes {full.coef_[0]:.3f} vs {wo.coef_[0]:.3f}; cook max {cook.max():.2f}, next {np.sort(cook)[-2]:.3f}")
    save(fig, "influence")


# --------------------------------------------------------------------------- security
def digits_models():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    lr = LogisticRegression(max_iter=3000).fit(X_tr, y_tr)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0).fit(X_tr, y_tr)
    rf = RandomForestClassifier(300, random_state=0).fit(X_tr, y_tr)
    return X_tr, X_te, y_tr, y_te, lr, mlp, rf


def fgsm(model, X, y, eps):
    P = model.predict_proba(X)
    P[np.arange(len(y)), y] -= 1.0                     # gradient of the log loss w.r.t. the logits
    grad = P @ model.coef_                              # chain rule through the linear layer
    return np.clip(X + eps * np.sign(grad), 0, 1)


def fig_adversarial():
    X_tr, X_te, y_tr, y_te, lr, mlp, rf = digits_models()
    adv = LogisticRegression(max_iter=3000).fit(np.r_[X_tr, fgsm(lr, X_tr, y_tr, 0.15)], np.r_[y_tr, y_tr])
    eps = np.array([0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3])
    rows = {"logistic regression (white box)": [], "neural network (transferred)": [],
            "random forest (transferred)": [], "adv. trained LR, attack on the old LR": [],
            "adv. trained LR, adaptive attack": []}
    for e in eps:
        Xa = fgsm(lr, X_te, y_te, e)
        rows["logistic regression (white box)"].append(lr.score(Xa, y_te))
        rows["neural network (transferred)"].append(mlp.score(Xa, y_te))
        rows["random forest (transferred)"].append(rf.score(Xa, y_te))
        rows["adv. trained LR, attack on the old LR"].append(adv.score(Xa, y_te))
        rows["adv. trained LR, adaptive attack"].append(adv.score(fgsm(adv, X_te, y_te, e), y_te))
    fig = plt.figure(figsize=(FULL, 2.45))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.6, 0.12, 0.8, 0.8, 0.8], wspace=0.25, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    for (name, v), c in zip(rows.items(), [NAVY, CORAL, GOLD, TEAL, INDIGO]):
        ax.plot(eps, v, "o-", ms=3, color=c, label=name)
        print(f"  {name}: " + " ".join(f"{a:.3f}" for a in v))
    ax.set_xlabel("FGSM perturbation $\\varepsilon$ (pixel scale 0-1)")
    ax.set_ylabel("test accuracy")
    ax.legend(loc="lower left", fontsize=5.6)
    ax.set_ylim(0, 1.02)
    i = int(np.flatnonzero((lr.predict(X_te) == y_te) & (lr.predict(fgsm(lr, X_te, y_te, 0.15)) != y_te))[0])
    xa = fgsm(lr, X_te[i:i + 1], y_te[i:i + 1], 0.15)[0]
    for col, (img, title) in enumerate([(X_te[i], f"original\npredicted {lr.predict(X_te[i:i + 1])[0]}"),
                                        (xa - X_te[i], "perturbation\n$\\pm0.15$"),
                                        (xa, f"adversarial\npredicted {lr.predict(xa[None])[0]}")]):
        a2 = fig.add_subplot(gs[:, 2 + col])
        a2.imshow(img.reshape(8, 8), cmap="gray_r" if col != 1 else "coolwarm",
                  vmin=0 if col != 1 else -0.2, vmax=1 if col != 1 else 0.2)
        a2.set_title(title, fontsize=6.8)
        a2.set_xticks([])
        a2.set_yticks([])
        a2.grid(False)
    save(fig, "adversarial_digits")


def spam_2d(n, seed):
    rng = np.random.default_rng(seed)
    Xh = rng.normal([-1.2, -1.2], 1.0, (n, 2))
    Xs = rng.normal([1.2, 1.2], 1.0, (n, 2))
    return np.r_[Xh, Xs], np.r_[np.zeros(n, int), np.ones(n, int)]


def fig_poisoning():
    X, y = spam_2d(1000, 0)
    Xt, yt = spam_2d(5000, 1)
    rng = np.random.default_rng(2)
    fracs = [0, 0.02, 0.05, 0.1, 0.15, 0.2]
    rows = []
    models = {}
    for fr in fracs:
        k = int(fr * len(y))
        chaff = np.random.default_rng(2).normal([1.0, 1.0], 0.35, (k, 2))   # nested chaff sets
        m = LogisticRegression().fit(np.r_[X, chaff], np.r_[y, np.zeros(k, int)])
        p = m.predict_proba(Xt)[:, 1]
        pred = p >= 0.5
        rows.append((fr, pred[yt == 1].mean(), pred[yt == 0].mean(), np.mean((p > 0.4) & (p < 0.6))))
        models[fr] = (m, chaff)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5))
    ax = axes[0]
    sub = np.r_[np.arange(0, 1000, 5), np.arange(1000, 2000, 5)]
    ax.scatter(*X[sub][y[sub] == 0].T, s=5, color=TEAL, label="ham")
    ax.scatter(*X[sub][y[sub] == 1].T, s=5, color=CORAL, label="spam")
    m10, chaff = models[0.1]
    ax.scatter(*chaff[::2].T, s=6, marker="x", color="black", lw=0.6, label="chaff (labelled ham)")
    xx = np.linspace(-4, 4, 2)
    for fr, ls, lab in [(0, "--", "clean model"), (0.1, "-", "poisoned (10% chaff)")]:
        m = models[fr][0]
        w, b = m.coef_[0], m.intercept_[0]
        ax.plot(xx, -(b + w[0] * xx) / w[1], ls, color=NAVY, lw=1.2, label=lab)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title("(a) A red-herring poisoning attack", fontsize=8)
    ax.legend(loc="lower right", fontsize=5.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[1]
    r = np.array(rows)
    ax.plot(r[:, 0] * 100, r[:, 1], "o-", ms=3, color=CORAL, label="spam detected (TPR)")
    ax.plot(r[:, 0] * 100, r[:, 2], "s-", ms=3, color=TEAL, label="ham flagged (FPR)")
    ax.plot(r[:, 0] * 100, r[:, 3], "^-", ms=3, color=GOLD, label="predictions with 0.4 < p < 0.6")
    ax.set_xlabel("chaff added to the training data (%)")
    ax.set_ylabel("proportion")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", fontsize=6.2)
    ax.set_title("(b) Effect on the detector", fontsize=8)
    for row in rows:
        print("  poison %.2f  TPR %.3f  FPR %.3f  band %.3f" % row)
    save(fig, "poisoning")


def emails(n, seed):
    rng = np.random.default_rng(seed)
    spam = np.r_[np.ones(n // 2, int), np.zeros(n - n // 2, int)]
    p = np.empty((n, 60))
    p[:, :15] = np.where(spam[:, None] == 1, 0.30, 0.03)     # "spammy" words
    p[:, 15:30] = np.where(spam[:, None] == 1, 0.05, 0.25)   # "hammy" words
    p[:, 30:] = 0.10                                         # neutral words
    return (rng.random((n, 60)) < p).astype(float), spam


def good_word_attack(model, x, budget):
    x = x.copy()
    for k in range(budget + 1):
        if model.predict_proba(x[None])[0, 1] < 0.5:
            return k
        if k == budget:
            break
        cands = np.flatnonzero(x == 0)
        Z = np.repeat(x[None], len(cands), axis=0)
        Z[np.arange(len(cands)), cands] = 1
        x[cands[np.argmin(model.predict_proba(Z)[:, 1])]] = 1
    return None


def fig_evasion():
    X, y = emails(4000, 0)
    Xt, yt = emails(2000, 1)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    spam_tr = X[y == 1][:600]
    adv_rows = []
    for x in spam_tr:
        xx = x.copy()
        for _ in range(10):
            if clf.predict_proba(xx[None])[0, 1] < 0.5:
                break
            cands = np.flatnonzero(xx == 0)
            Z = np.repeat(xx[None], len(cands), axis=0)
            Z[np.arange(len(cands)), cands] = 1
            xx[cands[np.argmin(clf.predict_proba(Z)[:, 1])]] = 1
        adv_rows.append(xx)
    robust = LogisticRegression(max_iter=2000).fit(np.r_[X, np.array(adv_rows)],
                                                   np.r_[y, np.ones(len(adv_rows), int)])
    budgets = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(HALF + 0.6, 2.4))
    for model, c, lab in [(clf, CORAL, "standard filter"), (robust, TEAL, "adversarially trained filter")]:
        caught = Xt[(yt == 1) & (model.predict(Xt) == 1)][:300]
        need = [good_word_attack(model, x, 10) for x in caught]
        succ = [np.mean([k is not None and k <= b for k in need]) for b in budgets]
        ax.plot(budgets, succ, "o-", ms=3, color=c, label=lab)
        print(f"  {lab}: detection {model.predict(Xt)[yt == 1].mean():.3f}, FPR {model.predict(Xt)[yt == 0].mean():.3f}, "
              f"success at 1/3/5/10 words: {succ[1]:.2f} {succ[3]:.2f} {succ[5]:.2f} {succ[10]:.2f}")
    ax.set_xlabel('"good" words the attacker may add')
    ax.set_ylabel("spam that evades the filter")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "evasion")


ALL = [fig_fairness_credit, fig_linear, fig_pdp_ice, fig_importance_h, fig_lime_shap,
       fig_counterfactual, fig_mmd, fig_influence, fig_adversarial, fig_poisoning, fig_evasion]

if __name__ == "__main__":
    use_style()
    only = set(sys.argv[2:])
    for f in ALL:
        if not only or f.__name__ in only:
            print(f.__name__)
            f()
