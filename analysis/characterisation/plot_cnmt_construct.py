#!/usr/bin/env python3
"""Plot the descriptive relationship between c_nmt and c_mono."""

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
INPUT = args.input
OUTPUT = args.output
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

BLUE = "#0072B2"
ORANGE = "#D55E00"
DEEP_BLUE = "#004C78"
DEEP_ORANGE = "#A53F00"
DARK_GREY = "#555555"
MID_GREY = "#888888"
LIGHT_GREY = "#AAAAAA"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

data = pd.read_csv(INPUT)
classes = [
    ("content", "Content words", BLUE, "o"),
    ("function", "Function words", ORANGE, "^"),
    ("other", "Other", MID_GREY, "s"),
]

fig, ax = plt.subplots(figsize=(6.65, 4.45), constrained_layout=True)

x_min, x_max = -1.75, 4.05
y_min, y_max = -1.35, 6.55
ax.add_patch(
    Rectangle(
        (x_min, 0),
        -x_min,
        y_max,
        facecolor=BLUE,
        edgecolor="none",
        alpha=0.035,
        zorder=0,
    )
)
ax.add_patch(
    Rectangle(
        (0, y_min),
        x_max,
        -y_min,
        facecolor=ORANGE,
        edgecolor="none",
        alpha=0.04,
        zorder=0,
    )
)

for group, label, colour, marker in classes:
    subset = data.loc[data["word_class"] == group]
    ax.scatter(
        subset["z_c_mono"],
        subset["z_c_nmt"],
        s=12 if group != "other" else 15,
        marker=marker,
        color=colour,
        alpha=0.18 if group != "other" else 0.24,
        linewidths=0,
        rasterized=True,
        label=f"{label} ($n={len(subset):,}$)",
        zorder=2,
    )

# Equal standardised values: points above this line are relatively higher on c_nmt.
diag_low = max(x_min, y_min)
diag_high = min(x_max, y_max)
ax.plot(
    [diag_low, diag_high],
    [diag_low, diag_high],
    color="#555555",
    linestyle=(0, (4, 4)),
    linewidth=0.9,
    alpha=0.7,
    zorder=1,
)
ax.axhline(0, color="#888888", linewidth=0.65, zorder=1)
ax.axvline(0, color="#888888", linewidth=0.65, zorder=1)

# Overall least-squares trend.
coef = np.polyfit(data["z_c_mono"], data["z_c_nmt"], 1)
grid = np.linspace(x_min, x_max, 200)
ax.plot(grid, np.polyval(coef, grid), color="#252525", linewidth=1.4, zorder=3)

labelled_examples = [
    ("wheelchair", "higher_nmt", (18, 8)),
    ("trucks", "higher_nmt", (-54, 13)),
    ("glasses", "higher_nmt", (17, -16)),
    ("jockey", "higher_mono", (16, 8)),
    ("parade", "higher_mono", (16, 10)),
    ("elephant", "higher_mono", (-52, -17)),
]
for word, direction, offset in labelled_examples:
    row = (
        data.loc[data["word"].str.lower().str.rstrip(".,;:!?") == word]
        .assign(
            divergence=lambda frame: (
                frame["z_c_nmt"] - frame["z_c_mono"]
                if direction == "higher_nmt"
                else frame["z_c_mono"] - frame["z_c_nmt"]
            )
        )
        .sort_values("divergence", ascending=False)
        .iloc[0]
    )
    colour = DEEP_BLUE if direction == "higher_nmt" else DEEP_ORANGE
    ax.scatter(
        row["z_c_mono"],
        row["z_c_nmt"],
        s=50,
        facecolor=colour,
        edgecolor="white",
        linewidth=0.8,
        alpha=1.0,
        zorder=5,
    )
    ax.annotate(
        word,
        (row["z_c_mono"], row["z_c_nmt"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.5,
        fontstyle="italic",
        color="#333333",
        arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.7},
        zorder=6,
    )

ax.text(
    x_min + 0.10,
    y_max - 0.30,
    "Translation-weighted divergence\n"
    "higher $c_{nmt}$ / lower $c_{mono}$",
    color=BLUE,
    fontsize=8.5,
    ha="left",
    va="top",
)
ax.text(
    x_max - 0.10,
    y_min + 0.10,
    "Monolingual divergence\n"
    "higher $c_{mono}$ / lower $c_{nmt}$",
    color=ORANGE,
    fontsize=8.5,
    ha="right",
    va="bottom",
)
ax.text(
    0.03,
    0.04,
    "$r=.541$ overall\n$r=.320$ after within-POS centring",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=8.5,
    color="#333333",
)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("Monolingual surprisal, $c_{mono}$ (SD units)")
ax.set_ylabel("Aligned NMT surprisal, $c_{nmt}$ (SD units)")
ax.grid(True, color="#E6E6E6", linewidth=0.7, zorder=0)
class_handles = [
    Line2D(
        [0], [0], marker=marker, linestyle="none", markersize=5,
        markerfacecolor=colour, markeredgewidth=0,
        label=f"{label} ($n={len(data.loc[data['word_class'] == group]):,}$)",
    )
    for group, label, colour, marker in classes
]
line_handles = [
    Line2D([0], [0], color="#252525", linewidth=1.4, label="Overall trend"),
    Line2D(
        [0], [0], color="#555555", linewidth=0.9,
        linestyle=(0, (4, 4)), label="Equal standardised values",
    ),
]
ax.legend(
    handles=class_handles + line_handles,
    title="Point classes and reference lines",
    loc="upper right",
    frameon=False,
    handletextpad=0.4,
    labelspacing=0.35,
    title_fontsize=8.7,
)

fig.savefig(OUTPUT, bbox_inches="tight")
print(OUTPUT)
