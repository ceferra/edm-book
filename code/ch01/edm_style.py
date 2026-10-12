"""Shared plotting style for the EDM course book.

The colours are taken from the theme of the course slides so that figures,
slides and book share the same visual identity.
"""
import matplotlib as mpl

NAVY, BRONZE, TEAL, CORAL = "#233A44", "#AF7B51", "#00796B", "#D9563F"
GOLD, INDIGO, GRAY, LIGHT = "#C4A15A", "#3D4594", "#8C8C8C", "#D9D9D9"
CYCLE = [NAVY, CORAL, TEAL, GOLD, INDIGO, BRONZE]

FULL = 6.0   # full text width (inches)
HALF = 3.3   # roughly half text width (inches)


def use_style():
    """Apply the book style to matplotlib."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    })
