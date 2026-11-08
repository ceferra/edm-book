# Evaluation, Deployment and Monitoring of Models

Course book of the Degree in Data Science (GCD), Universitat Politècnica de València.

Cèsar Ferri, Carlos Monserrat, Yael Moros, Félix Martí, Daniel Romero-Alvarado and Fernando Martínez-Plumed.

## Editions

| File | Edition |
|---|---|
| `main.tex` | Digital edition: A4, with links, bookmarks and the cover |
| `kdp.tex` | Print edition for Amazon KDP: 8.5 x 11 in, standard colour, no links |

Both are compiled with **XeLaTeX** (`latexmkrc` selects it, also on Overleaf):

```
latexmk main.tex      # digital edition
latexmk kdp.tex       # print edition
```

The PDF of the digital edition is attached to each
[release](https://github.com/ceferra/edm-book/releases). The workflow
`.github/workflows/book-pdf.yml` compiles it with TeX Live 2025 when a tag `v*` is pushed
(`git tag v1.1 && git push origin v1.1`), and also on demand from the Actions tab, where it
leaves the PDF as an artifact of the run.

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
- `add_index.py` and `index_terms.py`: the subject index. `python add_index.py` inserts the
  entries of the term list into the chapters; entries added by hand are kept.

## Rights

Copyright © 2026 the authors. All rights reserved (see the copyright page of the book).
The Libertinus fonts are distributed under the SIL Open Font License.
