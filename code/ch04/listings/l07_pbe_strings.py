import re

TOKENS = {"word": r"[A-Za-z]+", "number": r"\d+"}
# positions: first, second, third, last and penultimate token
INDEXES = (0, 1, 2, -1, -2)

def atoms(inp, piece):
    """The atoms of the DSL that produce the string `piece` from the input `inp`."""
    found = []
    for kind, rx in TOKENS.items():
        toks = re.findall(rx, inp)
        found += [("token", kind, k) for k in INDEXES
                  if -len(toks) <= k < len(toks) and toks[k] == piece]
    words = re.findall(TOKENS["word"], inp)
    if len(piece) == 1:
        found += [("initial", "word", k) for k in INDEXES
                  if -len(words) <= k < len(words) and words[k][0] == piece]
    if len(piece) <= 2 and not any(ch.isalnum() for ch in piece):
        found.append(("const", piece))            # language bias: short separators only
    return found

def run(program, inp):
    out = ""
    for kind, *arg in program:
        if kind == "const":
            out += arg[0]
            continue
        toks = re.findall(TOKENS[arg[0]], inp)
        if not -len(toks) <= arg[1] < len(toks):
            return None
        out += toks[arg[1]] if kind == "token" else toks[arg[1]][0]
    return out

def consistent(inp, out):
    """Every program (sequence of atoms) that maps inp to out."""
    if not out:
        return [()]
    return [(a,) + rest for i in range(1, len(out) + 1)
            for a in atoms(inp, out[:i]) for rest in consistent(inp, out[i:])]

def learn(examples):
    progs = [p for p in consistent(*examples[0])
             if all(run(p, i) == o for i, o in examples[1:])]
    return sorted(progs, key=lambda p: (len(p), sum(a[0] == "const" for a in p)))

show = lambda p: " + ".join(f'"{a[1]}"' if a[0] == "const" else f"{a[0]}({a[1]},{a[2]})"
                            for a in p)
tasks = {"first name": ([("Cesar Ferri", "Cesar"), ("Ana Gil", "Ana")],
                        ["Pau Torres", "Maria Jose Ramirez"]),
         "US date": ([("23/12/2021", "12-23-2021")], ["05/01/2022"]),
         "initials": ([("Cesar Ferri", "C.F."), ("Ana Gil", "A.G.")], ["Juan Garcia"]),
         "domain": ([("ana.gil@upv.es", "upv.es")], ["cferri@dsic.upv.es"])}
for name, (examples, tests) in tasks.items():
    progs = learn(examples)
    outs = {t: sorted({run(p, t) for p in progs}) for t in tests}
    print(f"{name}: {len(progs)} consistent programs")
    print(f"    best: {show(progs[0])}")
    for t, o in outs.items():
        print(f"    {t} -> {run(progs[0], t)}   (all consistent programs: {o})")
