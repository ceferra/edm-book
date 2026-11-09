"""Drops the repeated entries of the tables of contents of the web edition:
python3 html/fix_toc.py _site

The book lists its unnumbered chapters and sections (Preface, Bibliographic notes, Exercises...)
with \\chapter* or \\section* followed by \\addcontentsline, as LaTeX needs. TeX4ht, however,
already lists every starred heading by itself, so each of them would appear twice, both in the
contents of the first page and in the contents at the top of every chapter. The entry of the
starred heading is kept: it is the one that always points to the right page (the
\\addcontentsline of the bibliography is issued on the page before it).
"""
import os
import re
import sys

site = sys.argv[1]

# one entry: <span class='sectionToc'>1.2 <a href=...>Title</a></span>, up to the next entry
entry = re.compile(r"<span class='(\w+)Toc'>(.*?)</span>(?=\s*(?:<br />|</div>))", re.S)
separator = re.compile(r"\s*<br />\s*")


def title(body):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", body)).strip()


def fix(html):
    entries = list(entry.finditer(html))
    cut = []  # (start, end) of the repeated entries, with the <br /> before them
    for i, m in enumerate(entries):
        level, body = m.group(1), m.group(2)
        if level not in ("chapter", "section") or not re.match(r"\s*<a\b", body):
            continue  # only the unnumbered entries of \addcontentsline can be repeated
        for j in (i - 1, i + 1):
            if not 0 <= j < len(entries):
                continue
            other = entries[j]
            first, second = sorted((m, other), key=lambda e: e.start())
            if (other.group(1) == "like" + level and title(other.group(2)) == title(body)
                    and separator.fullmatch(html[first.end():second.start()])):
                start = m.start()
                before = html.rfind("<br />", 0, start)
                if before >= 0 and not html[before + 6:start].strip():
                    start = len(html[:before].rstrip())  # the line of the entry goes too
                cut.append((start, m.end()))
                break
    for start, end in reversed(cut):
        html = html[:start] + html[end:]
    return html, len(cut)


total = 0
for page in sorted(p for p in os.listdir(site) if p.endswith(".html")):
    path = os.path.join(site, page)
    with open(path, encoding="utf-8") as f:
        html = f.read()
    fixed, count = fix(html)
    if count:
        with open(path, "w", encoding="utf-8") as f:
            f.write(fixed)
        total += count
print("fix_toc: %d repeated contents entries dropped" % total)
