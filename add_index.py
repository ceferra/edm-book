"""Insert \\index entries into the assembled chapters (chapters/*.tex), from index_terms.py.

Policy: a term is indexed at its first occurrence in each section or subsection; terms that occur in
more than FREQUENT subsections are indexed only where they are defined (inside \\emph or \\textbf, page
in bold) and at their first occurrence in each chapter. Code, mathematics, floats, captions, headings
and command arguments are never touched. The chapters are the source of the book: run it on them at any time; entries added by hand are kept."""
import pathlib
import re

from index_terms import SEE, TERMS

ROOT = pathlib.Path(__file__).parent
FREQUENT = 15
SKIP_ENVS = {"pylisting", "pyoutput", "equation", "equation*", "align", "align*", "tikzpicture", "algorithm",
             "algorithmic", "figure", "figure*", "table", "table*", "subfigure", "cases", "lstlisting", "verbatim"}
MASK_CMDS = ("cite", "citep", "citet", "citealp", "citeauthor", "citeyear", "ref", "cref", "Cref", "eqref", "label",
             "url", "href", "texttt", "code", "includegraphics", "input", "index", "pageref", "autoref", "nameref",
             "caption", "section", "subsection", "subsubsection", "chapter", "paragraph", "hyperref", "textsc")


def form_regex(form):
    sensitive = any(c.isupper() for c in form)
    out = ""
    for part in re.split(r"(\s+|-+)", form):
        if not part:
            continue
        if part.isspace():
            out += r"(?:\s+|~)"
        elif set(part) == {"-"}:
            out += r"(?:-{1,3}|\s+)"
        else:
            out += re.escape(part)
    plural = r"(?:s|es)?" if form[-1].isalpha() else ""
    return re.compile(r"(?<![\w\\])" + out + plural + r"(?!\w)", 0 if sensitive else re.IGNORECASE)


COMPILED = [(key, [form_regex(f) for f in forms]) for key, forms in TERMS]


def balanced_end(s, i):
    """Index just after the brace group that opens at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and (j == 0 or s[j - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def protected(line):
    """Boolean mask of the characters that must not be indexed."""
    m = [False] * len(line)
    def mark(a, b):
        for k in range(a, min(b, len(line))):
            m[k] = True
    c = re.search(r"(?<!\\)%", line)
    if c:
        mark(c.start(), len(line))
    for rx in (r"(?<!\\)\$\$.*?(?<!\\)\$\$", r"(?<!\\)\$.*?(?<!\\)\$", r"\\\(.*?\\\)", r"\\\[.*?\\\]"):
        for x in re.finditer(rx, line):
            mark(x.start(), x.end())
    for x in re.finditer(r"\\(verb|lstinline)\*?(.)(.*?)\2", line):
        mark(x.start(), x.end())
    for x in re.finditer(r"\\(" + "|".join(MASK_CMDS) + r")\*?(\[[^\]]*\])?\{", line):
        mark(x.start(), balanced_end(line, x.end() - 1))
    for x in re.finditer(r"\\item\[[^\]]*\]", line):
        mark(x.start(), x.end())
    for x in re.finditer(r"\\[A-Za-z@]+", line):
        mark(x.start(), x.end())
    return m


def definitional_spans(line):
    spans = []
    for x in re.finditer(r"\\(emph|textbf)\{", line):
        spans.append((x.end(), balanced_end(line, x.end() - 1)))
    return spans


def blocks(path):
    """Paragraph blocks of indexable lines: [(unit, [(line_index, line, mask)])]."""
    lines = path.read_text(encoding="utf-8").split("\n")
    out, cur, unit, skip = [], [], 0, []
    def flush():
        if cur:
            out.append((unit, list(cur)))
            cur.clear()
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"\\(section|subsection)\*?[\[{]", s):
            flush(); unit += 1
            continue
        b = re.findall(r"\\begin\{([^}]+)\}", line)
        e = re.findall(r"\\end\{([^}]+)\}", line)
        for env in b:
            if env in SKIP_ENVS or skip:
                skip.append(env)
        for env in e:
            if skip and skip[-1] == env:
                skip.pop()
        if skip or b or e or s.startswith("%") or not s:
            flush()
            continue
        cur.append((i, line, protected(line)))
    flush()
    return out


def matches(block):
    """{term: [(line_index, column_after_match, definitional, start_offset)]}, matching across line breaks."""
    text = "\n".join(line for _, line, _ in block)
    mask, starts = [], []
    for k, (i, line, m) in enumerate(block):
        starts.append(len(mask))
        mask.extend(m)
        if k < len(block) - 1:
            mask.append(False)
    spans = definitional_spans(text)
    found = {}
    for t, (key, rxs) in enumerate(COMPILED):
        for rx in rxs:
            for x in rx.finditer(text):
                if any(mask[x.start():x.end()]):
                    continue
                k = max(n for n in range(len(starts)) if starts[n] <= x.end() - 1)
                d = any(a <= x.start() and x.end() <= b for a, b in spans)
                found.setdefault(t, []).append((block[k][0], x.end() - starts[k], d, x.start()))
    return found


AUTO_KEYS = {key for key, _ in TERMS}


def strip_automatic(text):
    """Remove the entries this script inserts; entries added by hand (other keys) are kept."""
    def keep(m):
        return "" if m.group(1) in AUTO_KEYS else m.group(0)
    return re.sub(r"\\index\{([^{}|]*)(?:\|[^{}]*)?\}", keep, text)


def main():
    files = sorted((ROOT / "chapters").glob("ch0*.tex"))
    for f in files:                       # chapters are the source: start from them without automatic entries
        f.write_text(strip_automatic(f.read_text(encoding="utf-8")), encoding="utf-8")
    units_of = {}
    cache = {}
    for f in files:
        rows = []
        for unit, block in blocks(f):
            ms = matches(block)
            rows.append((unit, ms))
            for t in ms:
                units_of.setdefault(t, set()).add((f.name, unit))
        cache[f] = rows
    frequent = {t for t, u in units_of.items() if len(u) > FREQUENT}
    total = 0
    for f in files:
        lines = f.read_text(encoding="utf-8").split("\n")
        inserts = {}
        seen_unit, seen_chapter = set(), set()
        for unit, ms in cache[f]:
            for t, occ in ms.items():
                key = TERMS[t][0]
                for i, end, d, _ in sorted(occ, key=lambda o: o[3]):
                    if t in frequent:
                        take = (d and (t, unit, "def") not in seen_unit) or t not in seen_chapter
                    else:
                        take = (t, unit) not in seen_unit
                    if not take:
                        continue
                    seen_chapter.add(t)
                    seen_unit.add((t, unit) if t not in frequent else (t, unit, "def") if d else (t, unit, "first"))
                    inserts.setdefault(i, []).append((end, "\\index{%s%s}" % (key, "|textbf" if d else "")))
                    break
        for i, items in inserts.items():
            line = lines[i]
            for end, text in sorted(items, key=lambda x: -x[0]):
                line = line[:end] + text + line[end:]
                total += 1
            lines[i] = line
        f.write_text("\n".join(lines), encoding="utf-8")
    see = "\n".join("\\index{%s|see{%s}}" % (a, b) for a, b in SEE)
    (ROOT / "chapters" / "index-see.tex").write_text("% cross-references of the index (generated by add_index.py)\n" + see + "\n")
    print(f"index entries inserted: {total}; terms found: {len(units_of)} of {len(TERMS)}; frequent terms: {len(frequent)}")
    missing = [TERMS[t][0] for t in range(len(TERMS)) if t not in units_of]
    if missing:
        print("terms not found:", missing)


if __name__ == "__main__":
    main()
