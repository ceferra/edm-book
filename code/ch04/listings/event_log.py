"""Synthetic event log of a compensation-request process (Unit 4)."""
import numpy as np
import pandas as pd

ACTIVITIES = {"a": "register request", "b": "examine thoroughly",
              "c": "examine casually", "d": "check ticket", "e": "decide",
              "f": "reinitiate request", "g": "pay compensation", "h": "reject request"}
HOURS = {"a": 1, "b": 20, "c": 6, "d": 40, "e": 8, "f": 4,     # mean duration
         "g": 24, "h": 2}                                       # of each activity
STAFF = {"a": ("Ana", "Luis"), "b": ("Marta",), "c": ("Marta", "Jorge"),
         "d": ("Ana", "Luis"), "e": ("Sara",), "f": ("Sara",), "g": ("Pablo",),
         "h": ("Pablo",)}

def simulate_log(n_cases=500, seed=0, noise=0.0):
    """Register, examine and check the ticket (in any order), decide, and then pay,
    reject or start over. With probability `noise` a case deviates from the process."""
    rng = np.random.default_rng(seed)
    rows = []
    for case in range(n_cases):
        trace = ["a"]
        while True:
            exam = "b" if rng.random() < 0.4 else "c"
            trace += [exam, "d"] if rng.random() < 0.5 else ["d", exam]   # concurrent
            trace.append("e")
            r = rng.random()
            if r < 0.2:
                trace.append("f")                      # start over
                continue
            trace.append("g" if r < 0.6 else "h")
            break
        if rng.random() < noise:                       # a deviation
            if rng.random() < 0.5:
                trace.remove("d")                      # the ticket is not checked
            else:
                i = trace.index("e")                   # the decision comes too early
                trace[i - 1], trace[i] = trace[i], trace[i - 1]
        start = pd.Timestamp("2026-01-05 08:00")
        t = start + pd.Timedelta(days=float(rng.uniform(0, 90)))
        for act in trace:
            t += pd.Timedelta(hours=float(rng.exponential(HOURS[act])))
            rows.append((f"c{case:04d}", ACTIVITIES[act], t, rng.choice(STAFF[act])))
    return pd.DataFrame(rows, columns=["case", "activity", "timestamp", "resource"])
