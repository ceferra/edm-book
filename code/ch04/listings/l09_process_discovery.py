from event_log import ACTIVITIES, simulate_log
from pm import alpha_miner

log = simulate_log(500, seed=0).sort_values(["case", "timestamp"])
code = {name: letter for letter, name in ACTIVITIES.items()}    # a = register request...
traces = log.groupby("case")["activity"].agg(lambda s: "".join(code[a] for a in s))
top = traces.value_counts().head(3)
print(f"{len(traces)} cases, {len(log)} events, {traces.nunique()} variants")
print("most frequent:", ", ".join(f"{v} ({n})" for v, n in top.items()))

# Performance: time between consecutive events of each directly-follows pair
log["next"] = log.groupby("case")["activity"].shift(-1)
gap = log.groupby("case")["timestamp"].shift(-1) - log["timestamp"]
log["hours"] = gap.dt.total_seconds() / 3600
perf = log.dropna(subset=["next"]).groupby(["activity", "next"])["hours"]
print(perf.agg(["count", "mean"]).sort_values("mean", ascending=False).head(3).round(1))
span = log.groupby("case")["timestamp"].agg(lambda s: (s.max() - s.min()).total_seconds())
print(f"mean throughput time: {span.mean() / 86400:.1f} days")

for A, B in alpha_miner(list(traces)):                          # discovered places
    print("place:", "".join(sorted(A)) or "start", "->", "".join(sorted(B)) or "end")
