# Evaluation, Deployment and Monitoring of Models -- course book (draft, Units 1-2)

Course book for *Evaluation, Deployment and Monitoring of Models* (3rd year, Degree in Data Science, UPV).

## Compile
- **Overleaf**: upload this zip as a new project; compiler pdfLaTeX; main document `main.tex`.
- **Local**: `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).

## Structure
- `main.tex` -- preamble (layout, colours from the course slides, boxes, listing styles), title page, document skeleton.
  Chapters are called *Units*; add new units with `\input{chapters/chNN-...}`.
- `frontmatter/preface.tex` -- course description, syllabus, practical sessions, how to use the book, recommended reading.
- `chapters/ch01-model-evaluation.tex` -- Unit 1, Model evaluation.
- `chapters/ch02-hybrid-models.tex` -- Unit 2, Hybrid models.
- `references.bib` -- bibliography (natbib, author-year, `plainnat`).
- `figures/chNN/` -- figures of each unit (vector PDF); `cd_results.txt` holds the benchmark tables.
- `code/chNN/make_figures.py` -- regenerates every figure of the unit (`python make_figures.py`); seeds are fixed.
  `code/ch01/edm_style.py` holds the shared plotting style.
- `code/chNN/listings/` -- the code listings shown in the book, with the outputs printed in the text (`*.out`).

Python packages: numpy, scipy, pandas, matplotlib, scikit-learn (1.8), imbalanced-learn (0.14).

## Writing conventions
- Boxes: `keyidea`, `pitfall`, `further`, `objectives`, and numbered `example`/`definition`
  (`\begin{example}{Title}{label}` -> `\cref{ex:label}`).
- Code: `\begin{pylisting}[label={lst:x}]{Title} ... \end{pylisting}` followed by `\begin{pyoutput} ... \end{pyoutput}`.
- Confusion matrices: rows = predicted class, columns = actual class (scikit-learn uses the transpose).
