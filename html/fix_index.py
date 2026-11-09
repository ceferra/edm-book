"""Rewrites the subject index of the web edition: python3 html/fix_index.py _site

TeX4ht links every index entry to an anchor (dx...) left where \\index was used, but in a book
split into pages those links lack the page, and their text is a running number of the entry,
which means nothing to a reader. Here every link gets its page and, as text, the number of the
section it points to (or the name of an unnumbered chapter). Repeated sections of one entry are
merged, keeping the bold of a main reference, and the few numbers without an anchor are dropped.
"""
import os
import re
import sys

site = sys.argv[1]


def read(name):
    with open(os.path.join(site, name), encoding="utf-8") as f:
        return f.read()


pages = [p for p in os.listdir(site) if p.endswith(".html")]

# page of every anchor
anchor_page = {}
for page in pages:
    for anchor in re.findall(r"id=['\"](dx[0-9-]+)['\"]", read(page)):
        anchor_page[anchor] = page

# locator of every page, from the contents on the first page: the section number, the chapter
# number for the opening page of a chapter, "Exercises (2)" for an unnumbered section of a
# numbered chapter, and the title of an unnumbered chapter for its pages
page_loc = {}
chapter = None
toc_entry = re.compile(r"<span class='(\w+)'>\s*([0-9.]*)\s*<a href='([^'#]+)[^']*'[^>]*>(.*?)</a>", re.S)
for cls, number, page, title in toc_entry.findall(read("index.html")):
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title)).strip()
    if cls in ("chapterToc", "likechapterToc"):
        chapter = number or title
        page_loc.setdefault(page, chapter)
    elif number:
        page_loc.setdefault(page, number)
    elif chapter and chapter[0].isdigit():
        page_loc.setdefault(page, "%s (%s)" % (title, chapter))
    else:
        page_loc.setdefault(page, chapter)


def title_of(page):
    m = re.search(r"<title>(.*?)</title>", read(page), re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else page


locator = re.compile(r",\s*(<strong>)?(?:<a href='#(dx[0-9-]+)'>\d+</a>|\d+)(</strong>)?")
item = re.compile(r"(<div class='index-(?:sub)*item'>)(.*?)((?:" + locator.pattern + r")+)"
                  r"((?:,\s*<em>see(?:\s+also)?</em>.*?)?\s*</div>)", re.S)


def rewrite(m):
    start, term, locators, end = m.group(1), m.group(2), m.group(3), m.group(m.lastindex)
    merged = {}  # locator text -> [href, bold]
    for bold, anchor, _ in locator.findall(locators):
        if not anchor or anchor not in anchor_page:
            continue  # a number without an anchor cannot be linked
        page = anchor_page[anchor]
        text = page_loc.get(page) or title_of(page)
        if text in merged:
            merged[text][1] = merged[text][1] or bool(bold)
        else:
            merged[text] = [page + "#" + anchor, bool(bold)]
    links = []
    for text, (href, bold) in merged.items():
        link = "<a href='%s'>%s</a>" % (href, text)
        links.append("<strong>%s</strong>" % link if bold else link)
    return start + term + "".join(", " + link for link in links) + end


index = read("Index.html")
fixed, count = item.subn(rewrite, index)
with open(os.path.join(site, "Index.html"), "w", encoding="utf-8") as f:
    f.write(fixed)
print("fix_index: %d entries rewritten, %d anchors on %d pages" % (count, len(anchor_page), len(pages)))
