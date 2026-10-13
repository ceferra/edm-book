"""Assemble chapters/*.tex from src/chNN/*.tex.

Lines of the form  %%LISTING name | title | label%%  are replaced by the code in
code/chNN/listings/name.py and its captured output name.out, so the book always
shows code and output that were actually executed.
"""
import pathlib
import re

root = pathlib.Path(__file__).resolve().parent
CHAPTERS = {"ch00": "ch00-introduction", "ch01": "ch01-model-evaluation", "ch02": "ch02-hybrid-models",
            "ch03": "ch03-reliable-models",
            "ch04": "ch04-learning-to-learn",
            "ch05": "ch05-deployment",
            "ch06": "ch06-monitoring",
            "ch07": "ch07-applications"}
pat = re.compile(r"^%%LISTING (\S+) \| (.+?) \| (\S+)%%\s*$")
for ch, outname in CHAPTERS.items():
    lst_dir = root / "code" / ch / "listings"
    lines = []
    for f in sorted((root / "src" / ch).glob("*.tex")):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = pat.match(line)
            if not m:
                lines.append(line)
                continue
            name, title, label = m.groups()
            title = re.sub(r"(?<!\\)%", r"\\%", title).replace("Table 1.4", r"\cref{tab:regex}")
            code = (lst_dir / f"{name}.py").read_text().rstrip("\n")
            out = (lst_dir / f"{name}.out").read_text().rstrip("\n")
            lines.append(f"\\begin{{pylisting}}[label={{{label}}}]{{{title}}}")
            lines += code.splitlines()
            lines.append("\\end{pylisting}")
            if out.strip():
                lines.append("\\begin{pyoutput}")
                lines += out.splitlines()
                lines.append("\\end{pyoutput}")
        lines.append("")
    (root / "chapters").mkdir(exist_ok=True)
    (root / "chapters" / f"{outname}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("assembled", outname, len(lines), "lines")


# the subject index: insert \index entries into the assembled chapters
import subprocess as _sp, sys as _sys
_sp.run([_sys.executable, str(pathlib.Path(__file__).parent / "add_index.py")], check=True)
