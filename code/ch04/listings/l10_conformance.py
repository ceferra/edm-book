from event_log import ACTIVITIES, simulate_log
from pm import alpha_miner, fitness, token_replay

code = {name: letter for letter, name in ACTIVITIES.items()}

def traces_of(log):
    log = log.sort_values(["case", "timestamp"])
    to_str = lambda s: "".join(code[a] for a in s)
    return list(log.groupby("case")["activity"].agg(to_str))

# the model is discovered from a clean log and checked against new cases
model = alpha_miner(traces_of(simulate_log(500, seed=0)))
for noise in (0.0, 0.1, 0.3):
    traces = traces_of(simulate_log(500, seed=1, noise=noise))  # new cases to check
    bad = [t for t in traces if token_replay(t, model)[2] > 0]  # missing tokens
    print(f"deviation rate {noise:.0%}: fitness {fitness(traces, model):.3f}, "
          f"{len(bad)} of {len(traces)} cases do not fit")
for t in bad[:2]:
    p, c, m, r = token_replay(t, model)
    print(f"case {t}: produced {p}, consumed {c}, missing {m}, remaining {r}")
