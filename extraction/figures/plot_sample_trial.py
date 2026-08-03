#!/usr/bin/env python3
"""Create a publication-ready sample-stimulus gaze figure from real EMMT data."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "extraction" / "eye_tracking"))

from extract_eye_measures import load_sentences, word_x_ranges
from gaze_line import extract_trial_bouts


FONT_PATH = (
    REPO_ROOT / "assets" / "fonts" / "FreeSansBold.ttf"
)
DISPLAY_FONT = FontProperties(fname=str(FONT_PATH), size=15)
COMPARISON_FONT = FontProperties(fname=str(FONT_PATH), size=7.4)
READ_BLUE = "#0072B2"
TRANSLATE_ORANGE = "#E69F00"


def _stage_colormaps(stage_color, name):
    if stage_color == READ_BLUE:
        light, dark = "#B9DCEC", "#003B73"
    else:
        light, dark = "#F9D99A", "#9A5700"
    order_cmap = LinearSegmentedColormap.from_list(
        f"{name}_order", [light, stage_color, dark]
    )
    heat_cmap = LinearSegmentedColormap.from_list(
        f"{name}_heat", ["#FFFFFF", light, stage_color]
    )
    return order_cmap, heat_cmap


def _duration_heatmap(bouts, model, x_limits, y_limits):
    x_grid = np.linspace(*x_limits, 800)
    y_grid = np.linspace(*y_limits, 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(xx)
    for bout in bouts:
        residual_y = bout.mean_y - model.predict(bout.mean_x)
        weight = min(bout.duration_ms, 1000.0)
        density += weight * np.exp(
            -0.5 * ((xx - bout.mean_x) / 24.0) ** 2
            -0.5 * ((yy - residual_y) / 13.0) ** 2
        )
    if density.max() > 0:
        density /= density.max()
    return density


def _draw_words(
    ax,
    words,
    ranges,
    y=0,
    *,
    color="#171717",
    fontproperties=DISPLAY_FONT,
    alpha=1.0,
    zorder=5,
    outline=False,
):
    for word, (x0, x1, _) in zip(words, ranges):
        label = ax.text(
            (x0 + x1) / 2,
            y,
            word,
            ha="center",
            va="center",
            color=color,
            fontproperties=fontproperties,
            alpha=alpha,
            zorder=zorder,
        )
        if outline:
            label.set_path_effects(
                [path_effects.withStroke(linewidth=1.5, foreground="white")]
            )


def _draw_fixed_window_comparison(
    ax,
    read_all,
    read_line,
    read_model,
    trans_all,
    trans_line,
    trans_model,
    words,
    ranges,
):
    left, right = ranges[0][0], ranges[-1][1]
    x_limits = (left - 42, right + 42)
    x_grid = np.linspace(*x_limits, 400)

    ax.axhspan(140, 260, color="#B8B8B8", alpha=0.22, zorder=0)
    ax.axhline(200, color="#7A7A7A", linestyle="--", linewidth=0.8, zorder=1)

    # The EMMT source-code window was centred on the sentence's nominal
    # display coordinate. Showing the sentence here makes the basis of that
    # choice explicit. The second copy below represents the same
    # stimulus in the vertically displaced coordinate system of this trial's
    # recorded fixations; the stimulus itself did not move on screen.
    _draw_words(
        ax,
        words,
        ranges,
        y=200,
        color="#666666",
        fontproperties=COMPARISON_FONT,
        alpha=0.78,
        zorder=2,
    )

    for model, color in (
        (read_model, READ_BLUE),
        (trans_model, TRANSLATE_ORANGE),
    ):
        predicted = np.array([model.predict(x) for x in x_grid])
        ax.fill_between(
            x_grid,
            predicted - model.half_width,
            predicted + model.half_width,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x_grid, predicted, color=color, linewidth=1.4, zorder=2)

    reference_x = (left + right) / 2
    recorded_y = np.mean(
        [read_model.predict(reference_x), trans_model.predict(reference_x)]
    )
    _draw_words(
        ax,
        words,
        ranges,
        y=recorded_y,
        color="#252525",
        fontproperties=COMPARISON_FONT,
        alpha=0.88,
        zorder=4,
        outline=True,
    )

    ax.scatter(
        [bout.mean_x for bout in read_line],
        [bout.mean_y for bout in read_line],
        s=14,
        marker="o",
        color=READ_BLUE,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.62,
        label="Oral reading",
        zorder=3,
    )
    ax.scatter(
        [bout.mean_x for bout in trans_line],
        [bout.mean_y for bout in trans_line],
        s=16,
        marker="D",
        color=TRANSLATE_ORANGE,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.42,
        zorder=3,
    )

    accepted = {
        ("read", bout.sequence) for bout in read_line
    } | {
        ("translate", bout.sequence) for bout in trans_line
    }
    rejected = [
        bout
        for stage, bouts in (("read", read_all), ("translate", trans_all))
        for bout in bouts
        if (stage, bout.sequence) not in accepted
        and bout.mean_x is not None
        and bout.mean_y is not None
        and x_limits[0] <= bout.mean_x <= x_limits[1]
    ]
    if rejected:
        ax.scatter(
            [bout.mean_x for bout in rejected],
            [bout.mean_y for bout in rejected],
            marker="x",
            s=20,
            linewidth=0.8,
            color="#777777",
            alpha=0.55,
            zorder=2,
        )

    accepted_y = [bout.mean_y for bout in list(read_line) + list(trans_line)]
    lower = min(120, min(accepted_y) - 18)
    upper = max(365, max(accepted_y) + 30)
    ax.set_xlim(x_limits)
    ax.set_ylim(upper, lower)
    ax.set_ylabel("vertical screen coordinate (px)", fontsize=8)
    ax.set_xlabel("horizontal screen coordinate (px)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7, length=2)
    ax.text(
        x_limits[0] + 10,
        150,
        "EMMT source-code window around the nominal text line (200 ± 60 px)",
        fontsize=7.2,
        color="#5F5F5F",
        va="top",
        zorder=4,
    )
    offset_x = x_limits[0] + 26
    ax.annotate(
        "",
        xy=(offset_x, recorded_y),
        xytext=(offset_x, 215),
        arrowprops={
            "arrowstyle": "<->",
            "color": "#666666",
            "linewidth": 0.8,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        zorder=5,
    )
    ax.text(
        offset_x + 8,
        (215 + recorded_y) / 2,
        f"recorded offset\n{recorded_y - 200:.0f} px",
        fontsize=6.7,
        color="#444444",
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        zorder=5,
    )
    ax.text(
        x_limits[1] - 8,
        137,
        "oral reading",
        fontsize=6.8,
        color=READ_BLUE,
        ha="right",
        va="center",
        zorder=5,
    )
    ax.text(
        x_limits[1] - 8,
        153,
        "sight translation",
        fontsize=6.8,
        color=TRANSLATE_ORANGE,
        ha="right",
        va="center",
        zorder=5,
    )
    ax.text(
        x_limits[1] - 8,
        recorded_y + 21,
        "same sentence, line inferred from this trial's recorded gaze",
        fontsize=6.7,
        color="#333333",
        ha="right",
        va="top",
        zorder=5,
    )
    ax.set_title(
        "b  EMMT fixed window versus this trial's recorded gaze",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        pad=4,
    )
    for spine in ax.spines.values():
        spine.set_color("#b9b9b9")
        spine.set_linewidth(0.6)


def _draw_stage_panel(
    ax, all_bouts, line_bouts, model, words, ranges, title, stage_color, name
):
    left, right = ranges[0][0], ranges[-1][1]
    x_limits = (left - 42, right + 42)
    y_limits = (-72, 72)
    density = _duration_heatmap(line_bouts, model, x_limits, y_limits)
    order_cmap, heat_cmap = _stage_colormaps(stage_color, name)
    is_translation = name == "translate"
    ax.set_facecolor("#FFFFFF")
    ax.imshow(
        density,
        extent=(*x_limits, *y_limits),
        origin="lower",
        aspect="auto",
        cmap=heat_cmap,
        alpha=0.42 if is_translation else 0.58,
        vmin=0,
        vmax=1,
        zorder=1,
    )

    line_sequences = {bout.sequence for bout in line_bouts}
    rejected = [
        bout
        for bout in all_bouts
        if bout.sequence not in line_sequences
        and bout.mean_x is not None
        and bout.mean_y is not None
        and x_limits[0] <= bout.mean_x <= x_limits[1]
    ]
    if rejected:
        ax.scatter(
            [bout.mean_x for bout in rejected],
            [bout.mean_y - model.predict(bout.mean_x) for bout in rejected],
            marker="x",
            s=22,
            linewidth=0.8,
            color="#737373",
            alpha=0.45,
            zorder=2,
        )

    ordered = sorted(line_bouts, key=lambda bout: bout.sequence)
    xs = np.array([bout.mean_x for bout in ordered])
    ys = np.array([bout.mean_y - model.predict(bout.mean_x) for bout in ordered])
    order = np.arange(len(ordered))
    if len(ordered) > 1:
        points = np.column_stack([xs, ys])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lines = LineCollection(
            segments,
            cmap=order_cmap,
            norm=Normalize(0, max(len(ordered) - 1, 1)),
            linewidth=1.15,
            alpha=0.76 if is_translation else 0.88,
            zorder=3,
        )
        lines.set_array(order[:-1])
        ax.add_collection(lines)
    sizes = 20 + 75 * np.sqrt(
        np.array([bout.duration_ms for bout in ordered])
        / max(bout.duration_ms for bout in ordered)
    )
    ax.scatter(
        xs,
        ys,
        c=order,
        cmap=order_cmap,
        s=sizes,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.68 if is_translation else 0.90,
        zorder=4,
    )
    for index in range(0, len(ordered), 5):
        ax.text(
            xs[index],
            ys[index],
            str(index + 1),
            ha="center",
            va="center",
            fontsize=5.5,
            color="white",
            fontweight="bold",
            zorder=6,
        )

    _draw_words(ax, words, ranges)
    ax.axhline(0, color="white", linewidth=0.45, alpha=0.55, zorder=2)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_yticks([-45, 0, 45])
    ax.set_ylabel("vertical residual (px)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7, length=2)
    ax.set_xlabel("horizontal screen coordinate (px)", fontsize=8)
    ax.set_title(
        f"{title}: {len(line_bouts)} mapped fixation bouts",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        pad=4,
    )
    for spine in ax.spines.values():
        spine.set_color("#b9b9b9")
        spine.set_linewidth(0.6)


def _serialise_stage(all_bouts, line_bouts, model):
    accepted = {bout.sequence for bout in line_bouts}
    return {
        "model": {
            "alpha": model.alpha,
            "beta": model.beta,
            "x_center": model.x_center,
            "half_width": model.half_width,
            "fit_source": model.fit_source,
        },
        "bouts": [
            {
                "sequence": bout.sequence,
                "x": bout.mean_x,
                "y": bout.mean_y,
                "residual_y": (
                    bout.mean_y - model.predict(bout.mean_x)
                    if bout.mean_x is not None and bout.mean_y is not None
                    else None
                ),
                "duration_ms": bout.duration_ms,
                "accepted": bout.sequence in accepted,
            }
            for bout in all_bouts
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_file", required=True)
    parser.add_argument("--translate_file", required=True)
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--sentence_id", required=True)
    parser.add_argument("--output", required=True, help="PNG or PDF output path")
    parser.add_argument("--json_output", help="Optional data file for interactive display")
    args = parser.parse_args()

    sentences = load_sentences(args.sentences)
    words = sentences[args.sentence_id]
    ranges = word_x_ranges(words)
    read_all, read_line, read_diagnostic, read_model = extract_trial_bouts(
        args.read_file, ranges, stage="read"
    )
    if read_model is None:
        raise RuntimeError(f"READ line rejected: {read_diagnostic.reason}")
    trans_all, trans_line, trans_diagnostic, trans_model = extract_trial_bouts(
        args.translate_file,
        ranges,
        stage="translate",
        prior_model=read_model,
    )
    if trans_model is None:
        raise RuntimeError(f"TRANSLATE line rejected: {trans_diagnostic.reason}")

    fig = plt.figure(figsize=(8.0, 7.5), facecolor="white")
    grid = fig.add_gridspec(
        4, 1, height_ratios=[0.54, 1.24, 1, 1], hspace=0.52
    )
    stimulus_ax = fig.add_subplot(grid[0])
    stimulus_ax.set_facecolor("#eeeeeb")
    _draw_words(stimulus_ax, words, ranges)
    stimulus_ax.set_xlim(ranges[0][0] - 42, ranges[-1][1] + 42)
    stimulus_ax.set_ylim(-26, 26)
    stimulus_ax.set_xticks([])
    stimulus_ax.set_yticks([])
    stimulus_ax.set_title(
        f"a  Recorded source-text stimulus ({args.sentence_id})",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        pad=5,
    )
    for spine in stimulus_ax.spines.values():
        spine.set_color("#c9c9c5")
        spine.set_linewidth(0.7)

    comparison_ax = fig.add_subplot(grid[1])
    _draw_fixed_window_comparison(
        comparison_ax,
        read_all,
        read_line,
        read_model,
        trans_all,
        trans_line,
        trans_model,
        words,
        ranges,
    )

    read_ax = fig.add_subplot(grid[2])
    translate_ax = fig.add_subplot(grid[3])
    _draw_stage_panel(
        read_ax,
        read_all,
        read_line,
        read_model,
        words,
        ranges,
        "c  Oral reading",
        READ_BLUE,
        "read",
    )
    _draw_stage_panel(
        translate_ax,
        trans_all,
        trans_line,
        trans_model,
        words,
        ranges,
        "d  Sight translation",
        TRANSLATE_ORANGE,
        "translate",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.97, bottom=0.08)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if args.json_output:
        payload = {
            "sentence_id": args.sentence_id,
            "sentence": " ".join(words),
            "words": words,
            "ranges": [list(item[:2]) for item in ranges],
            "read": _serialise_stage(read_all, read_line, read_model),
            "translate": _serialise_stage(trans_all, trans_line, trans_model),
        }
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
