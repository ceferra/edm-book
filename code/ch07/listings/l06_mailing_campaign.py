import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from crm_data import pilot

X_tr, y_tr, X_val, y_val, X_rest, p_rest, rng = pilot()
print(f"pilot: {len(y_tr) + len(y_val):,} customers mailed, "
      f"response rate {np.r_[y_tr, y_val].mean():.1%}")
model = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
score = model.predict_proba(X_val)[:, 1]
print(f"validation AUC {roc_auc_score(y_val, score):.3f}")
ranked = y_val[np.argsort(-score)]                     # responses in mailing order
gains = np.cumsum(ranked) / ranked.sum()               # share of all responses captured
for f in (0.1, 0.2, 0.5):
    n = int(f * len(ranked))
    print(f"mailing the top {f:.0%}: {gains[n - 1]:.0%} of the responses, "
          f"lift {gains[n - 1] / f:.1f}")

# the campaign: the rest of the base
N = len(p_rest)
share = np.arange(1, len(ranked) + 1) / len(ranked)
CONFIGS = {"config 1": (10_000, 1.50, 30.0), "config 2": (20_000, 0.80, 25.0)}
curves = {}
order_rest = np.argsort(-model.predict_proba(X_rest)[:, 1])   # the real campaign
for name, (fixed, per_mail, margin) in CONFIGS.items():
    mailed = margin * np.cumsum(ranked) * N / len(ranked) - per_mail * share * N - fixed
    profit = np.r_[0.0, mailed]                        # index 0: do not launch at all
    curves[name] = profit
    best = int(np.argmax(profit))
    if best == 0:
        print(f"{name}: do not launch the campaign")
        continue
    n_mail = int(best / len(ranked) * N)
    real = margin * p_rest[order_rest[:n_mail]].sum() - per_mail * n_mail - fixed
    print(f"{name}: mail the top {best / len(ranked):.0%} ({n_mail:,} customers), "
          f"{profit[best]:,.0f} EUR on validation")
    print(f"          true expected profit in the simulation {real:,.0f} EUR; "
          f"mailing everyone {profit[-1]:,.0f} EUR")
