"""Process mining from scratch: directly-follows, alpha algorithm and token replay."""
from collections import Counter
from itertools import combinations

def directly_follows(traces):
    """Frequency of each pair (a, b) such that b directly follows a in some trace."""
    return Counter((a, b) for t in traces for a, b in zip(t, t[1:]))

def alpha_miner(traces, max_size=None):
    """Alpha algorithm (van der Aalst et al., 2004). Returns the places of a Petri net
    as (input transitions, output transitions), with a source and a sink place."""
    df = set(directly_follows(traces))
    acts = sorted({a for t in traces for a in t})
    causal = lambda a, b: (a, b) in df and (b, a) not in df          # a -> b
    unrelated = lambda a, b: (a, b) not in df and (b, a) not in df   # a # b
    limit = len(acts) if max_size is None else max_size   # None: the classical algorithm
    sets = [frozenset(s) for k in range(1, limit + 1) for s in combinations(acts, k)
            if all(unrelated(a, b) for a in s for b in s)]
    pairs = [(A, B) for A in sets for B in sets
             if all(causal(a, b) for a in A for b in B)]
    maximal = [(A, B) for A, B in pairs
               if not any(A <= C and B <= D and (A, B) != (C, D) for C, D in pairs)]
    start, end = frozenset(t[0] for t in traces), frozenset(t[-1] for t in traces)
    return [(frozenset(), start)] + maximal + [(end, frozenset())]

def token_replay(trace, places):
    """Replay a trace on the net: tokens (produced, consumed, missing, remaining)."""
    unknown = set(trace) - set().union(*(A | B for A, B in places))
    if unknown:                                      # not modelled: report, never ignore
        raise ValueError(f"activities not in the model: {sorted(unknown)}")
    tokens = Counter({0: 1})                         # one token in the source place
    produced, consumed, missing = 1, 0, 0
    for act in trace:
        for i, (A, B) in enumerate(places):
            if act in B:                             # an input place of the transition
                if tokens[i] == 0:
                    missing += 1
                    tokens[i] += 1
                tokens[i] -= 1
                consumed += 1
        for i, (A, B) in enumerate(places):
            if act in A:                             # an output place
                tokens[i] += 1
                produced += 1
    sink = len(places) - 1
    if tokens[sink] == 0:
        missing += 1
        tokens[sink] += 1
    tokens[sink] -= 1
    consumed += 1
    return produced, consumed, missing, sum(tokens.values())

def fitness(traces, places):
    """Log fitness: (1 - missing/consumed) / 2 + (1 - remaining/produced) / 2."""
    p, c, m, r = map(sum, zip(*(token_replay(t, places) for t in traces)))
    return 0.5 * (1 - m / c) + 0.5 * (1 - r / p)
