#!/bin/sh
# Builds the web edition of the book into _site/ with TeX4ht (make4ht).
#
# Run it from the root of a scratch copy of the repository: it writes an .svg and an .xbb next
# to every PDF figure, and the auxiliary files of TeX4ht in the root. It needs TeX Live 2026 (the
# TeX4ht of TeX Live 2025 loses the MathML markup after the first TikZ picture) and poppler-utils:
#   docker run --rm -v "$PWD":/w -w /w texlive/texlive:latest sh -c \
#     'apt-get update && apt-get install -y poppler-utils && sh html/build.sh'
set -eu
out=_site

# 1. Figures. Browsers cannot show PDF images, so every PDF figure gets an SVG twin; TeX4ht
#    reads the size of each figure from the bounding box of its PDF (.xbb).
#    Scatter plots make SVGs of several MB: those go as PNG at 200 dpi instead.
find figures -name '*.pdf' ! -path 'figures/cover/*' | while read -r pdf; do
  base=${pdf%.pdf}
  pdftocairo -svg "$pdf" "$base.svg"
  if [ "$(wc -c < "$base.svg")" -gt 600000 ]; then
    pdftocairo -png -r 200 -singlefile "$pdf" "$base"
    rm "$base.svg"
  fi
  extractbb "$pdf"
done

# 2. HTML: a page per chapter and per section, mathematics as MathML rendered by MathJax.
rm -rf "$out"
cp html/edm.css edm.css
make4ht -l -a warning -j index -c html/edm.cfg -e html/build.mk4 -d "$out" \
  web.tex "3,sec-filename,mathml,mathjax,fn-in"

# Pages are named after their headings. Give the bibliography and the index readable names, and
# drop the "extsec" that the star of the extension sections (\extsec) leaves in some names.
cd "$out"
renames="bibnamemkbothMakeUppercasebibnameMakeUppercasebibname:Bibliography indexname:Index"
for f in *extsec.html; do
  [ -f "$f" ] && renames="$renames ${f%.html}:${f%extsec.html}"
done
for pair in $renames; do
  old=${pair%%:*} new=${pair#*:}
  if [ -f "$old.html" ] && [ ! -e "$new.html" ]; then
    mv "$old.html" "$new.html"
    sed -i "s/\b$old\.html/$new.html/g" ./*.html
  fi
done
# the "Up" links point to an anchor that TeX4ht does not create (#Page.html): keep the page only
sed -i "s/\(href='[^'#]*\.html\)#[^'#]*\.html'/\1'/g" ./*.html
cd - > /dev/null
# contents: TeX4ht lists \chapter* and \section* by itself, so their \addcontentsline repeats them
python3 html/fix_toc.py "$out"
# index entries: links to the right page, with section numbers instead of running numbers
python3 html/fix_index.py "$out"

# 3. Fonts and cover.
mkdir -p "$out/fonts"
cp fonts/LibertinusSerif-*.otf fonts/LibertinusSans-*.otf fonts/OFL.txt "$out/fonts/"
# the front cover is compiled from cover.tex (twice: its text is placed relative to the page)
(cd figures/cover && for pass in 1 2; do xelatex -interaction=batchmode -jobname=webcover cover.tex > /dev/null; done)
pdftocairo -jpeg -jpegopt quality=85 -singlefile -scale-to 900 figures/cover/webcover.pdf "$out/cover"
touch "$out/.nojekyll"
