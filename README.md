# Evaluation, Deployment and Monitoring of Models

Course book of the Degree in Data Science (GCD), Universitat Politècnica de València.

Cèsar Ferri, Carlos Monserrat, Yael Moros, Félix Martí, Daniel Romero-Alvarado and Fernando Martínez-Plumed.

**Read it online: <https://ceferra.github.io/edm-book/>** · PDF: [latest release](https://github.com/ceferra/edm-book/releases/latest)

## Editions

| File | Edition |
|---|---|
| `main.tex` | Digital edition: A4, with links, bookmarks and the cover |
| `kdp.tex` | Print edition for Amazon KDP: 8.5 x 11 in, standard colour, no links |
| `web.tex` | Web edition (HTML): a page per section, built with TeX4ht |

The two PDF editions are compiled with **XeLaTeX** (`latexmkrc` selects it, also on Overleaf):

```
latexmk main.tex      # digital edition
latexmk kdp.tex       # print edition
```

The PDF of the digital edition is attached to each
[release](https://github.com/ceferra/edm-book/releases). The workflow
`.github/workflows/book-pdf.yml` compiles it with TeX Live 2025 when a tag `v*` is pushed
(`git tag v1.1 && git push origin v1.1`), and also on demand from the Actions tab, where it
leaves the PDF as an artifact of the run.

The web edition is published with GitHub Pages by `.github/workflows/book-web.yml`, on the same
tags and also on demand. It is built from the same sources by `html/build.sh` (TeX Live 2026,
make4ht): the figures are converted from PDF to SVG, the formulas go as MathML rendered by
MathJax, and `html/edm.cfg` (TeX4ht configuration), `html/edm.css` (style),
`html/build.mk4` and `html/fix_index.py` adapt the boxes, listings, cover, bibliography and
subject index. To build it locally, in a scratch copy of the repository:

```
docker run --rm -v "$PWD":/w -w /w texlive/texlive:latest sh -c \
  'apt-get update && apt-get install -y poppler-utils && sh html/build.sh'   # -> _site/
```

## Structure

- `chapters/`: the text of the book. **These files are the source**: edit them directly.
- `frontmatter/`: preface. `references.bib`: bibliography.
- `code/chNN/listings/`: every listing as a Python file, with its output (`.out`); the
  versions used are pinned in `code/requirements.txt`. Each output printed in the book is the
  real output of its listing.
- `figures/`: figures of the book; `figures/cover/`: generative cover (`generar.py`,
  `make_cover.py`) and the full-wrap KDP cover (`make_kdp_cover.py`, which reads the page
  count of `kdp.pdf` and computes the spine).
- `fonts/`: Libertinus (SIL Open Font License, `fonts/OFL.txt`).
- `html/` and `web.tex`: the web edition (see above).
- `add_index.py` and `index_terms.py`: the subject index. `python add_index.py` inserts the
  entries of the term list into the chapters; entries added by hand are kept.

## Rights

Copyright © 2026 the authors. All rights reserved (see the copyright page of the book).
The Libertinus fonts are distributed under the SIL Open Font License.
