"""Book cover: the '02_flujo' artwork of generar.py (seed 42), cropped to A4 at the bottom,
with the title and the authors set in DejaVu Sans as in the original design.
Run: python make_cover.py   ->  cover-art.svg/.pdf (artwork only, vector, 210 x 297 mm), cover.png (preview)
The title and the authors are set by LaTeX in the book's fonts (main.tex, \\coverlines)."""
import random
import cairosvg
import generar

W, H = generar.W, round(generar.W * 297 / 210)        # 1600 x 2263: A4 proportions
TITLE = ("Evaluation, Deployment", "and Monitoring of Models")
AUTHORS = ("C\u00e8sar Ferri, Carlos Monserrat, Yael Moros, F\u00e9lix Mart\u00ed,", "Daniel Romero-Alvarado and Fernando Mart\u00ednez-Plumed")

background, paths = generar.flow(random.Random(42))
text = (f'<g font-family="DejaVu Sans" fill="#F5F2E9" font-weight="bold" font-size="78">'
        f'<text x="129" y="153">{TITLE[0]}</text><text x="129" y="252">{TITLE[1]}</text></g>'
        f'<g font-family="DejaVu Sans" font-size="34.5" fill="#85B3BC"><text x="129" y="364">{AUTHORS[0]}</text>'
        f'<text x="129" y="414">{AUTHORS[1]}</text></g>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 {W} {H}">'
       f'<rect width="100%" height="100%" fill="{background}"/>' + "".join(paths) + text + "</svg>")
open("cover.svg", "w").write(svg)
art = svg.replace(text, "")                               # the artwork alone, for LaTeX
open("cover-art.svg", "w").write(art)
cairosvg.svg2pdf(bytestring=art.encode(), write_to="cover-art.pdf")
cairosvg.svg2png(bytestring=svg.encode(), write_to="cover.png", output_width=1240, output_height=1754)
print("cover-art.svg, cover-art.pdf and cover.png written")
