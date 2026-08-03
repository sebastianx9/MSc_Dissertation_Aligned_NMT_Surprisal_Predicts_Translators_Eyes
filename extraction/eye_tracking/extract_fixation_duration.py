"""
Extract word-level total fixation duration for READ and TRANSLATE stages
from EMMT preprocessed gaze CSV files.

Word x-positions are computed from the experiment's Free Sans Bold font at
28 px.

Output: one row per participant x sentence x stage x word
"""

import argparse
import os
import csv
from pathlib import Path
from PIL import ImageFont
from gaze_line import (
    extract_trial_bouts,
    map_trial_bouts,
    x_to_word_index as map_x_to_word_index,
)

# Display parameters from experiment script
TEXT_CENTER_X = 620
FONT_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "fonts" / "FreeSansBold.ttf"
)
if not FONT_PATH.exists():
    raise FileNotFoundError(f"Required experimental font not found: {FONT_PATH}")
FONT = ImageFont.truetype(str(FONT_PATH), 28)
SPACE_WIDTH = float(FONT.getlength(" "))


def word_pixel_width(word):
    bbox = FONT.getbbox(word)
    return bbox[2] - bbox[0]


def load_sentences(path):
    """Return {sentence_id: [word1, word2, ...]} from Sentences.csv."""
    sentences = {}
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, text = line.split(",", 1)
            sentences[sid.strip()] = text.strip().split()
    return sentences


def word_x_ranges(words):
    """
    Return list of (x_start, x_end, word) for each word,
    using actual font pixel widths centered at TEXT_CENTER_X.
    """
    widths = [word_pixel_width(w) for w in words]
    total_width = sum(widths) + SPACE_WIDTH * (len(words) - 1)
    x = TEXT_CENTER_X - total_width / 2
    ranges = []
    for w, w_width in zip(words, widths):
        ranges.append((x, x + w_width, w))
        x += w_width + SPACE_WIDTH
    return ranges


def x_to_word_index(x, ranges):
    """
    Return index of the word whose x-range contains x.
    Falls back to the nearest word only when x falls in an inter-word space.
    Returns -1 outside the sentence's horizontal bounds.
    """
    return map_x_to_word_index(x, ranges)


def extract_word_fixations(
    filepath,
    words,
    stage="read",
    prior_model=None,
    return_diagnostic=False,
):
    """
    Return {word_index: total_fixation_ms} for one gaze CSV file.
    Groups consecutive fixation samples into bouts, estimates the trial's
    dominant text-line band, and maps retained bouts by mean x-position.
    """
    ranges = word_x_ranges(words)
    all_bouts, line_bouts, diagnostic, model = extract_trial_bouts(
        filepath,
        ranges,
        stage=stage,
        prior_model=prior_model,
    )
    mapped_bouts = map_trial_bouts(
        all_bouts, line_bouts, ranges, diagnostic=diagnostic
    )
    word_totals = {}
    for wi, duration_ms in mapped_bouts:
        if wi >= 0:
            word_totals[wi] = word_totals.get(wi, 0.0) + duration_ms
    if return_diagnostic:
        return word_totals, diagnostic, model
    return word_totals


def parse_filename(fname):
    if not fname.endswith(".csv"):
        return None
    parts = fname[:-4].split("-")
    if len(parts) != 5:
        return None
    return parts  # [participant, order, sentence_id, ambiguity, congruency]


def process_directory(
    directory,
    stage_label,
    sentences,
    results,
    diagnostics,
    read_line_models,
):
    for fname in sorted(os.listdir(directory)):
        parts = parse_filename(fname)
        if parts is None:
            continue
        participant, order, sentence_id, ambiguity, congruency = parts
        words = sentences.get(sentence_id)
        if words is None:
            continue
        fpath = os.path.join(directory, fname)
        trial_key = (participant, order, sentence_id)
        prior_model = (
            read_line_models.get(trial_key)
            if stage_label == "translate"
            else None
        )
        word_totals, diagnostic, model = extract_word_fixations(
            fpath,
            words,
            stage=stage_label,
            prior_model=prior_model,
            return_diagnostic=True,
        )
        if stage_label == "read" and model is not None:
            read_line_models[trial_key] = model
        diagnostics.append({
            "participant": participant,
            "order": order,
            "sentence_id": sentence_id,
            "ambiguity": ambiguity,
            "congruency": congruency,
            "stage": stage_label,
            "file": fname,
            **diagnostic.to_dict(),
        })
        for wi, dur in sorted(word_totals.items()):
            results.append({
                "participant":               participant,
                "order":                     order,
                "sentence_id":              sentence_id,
                "ambiguity":                ambiguity,
                "congruency":               congruency,
                "stage":                    stage_label,
                "word_index":               wi,
                "word":                     words[wi],
                "line_fit_source":          model.fit_source,
                "total_fixation_duration_ms": round(dur, 2),
            })


def remove_outliers(results, sd_threshold=2.5):
    """Remove rows where fixation duration > 2.5 SD from participant mean per stage."""
    from collections import defaultdict
    import math

    # group durations by (participant, stage)
    groups = defaultdict(list)
    for r in results:
        groups[(r["participant"], r["stage"])].append(r["total_fixation_duration_ms"])

    # compute mean and SD per group
    stats = {}
    for key, vals in groups.items():
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[key] = (mean, math.sqrt(variance))

    before = len(results)
    filtered = []
    for r in results:
        mean, sd = stats[(r["participant"], r["stage"])]
        if abs(r["total_fixation_duration_ms"] - mean) <= sd_threshold * sd:
            filtered.append(r)

    print(f"  Removed {before - len(filtered)} outlier rows out of {before} total.")
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Extract word-level TFD from EMMT gaze files.")
    parser.add_argument("--read_dir",      required=True,
                        help="Path to EMMT preprocessed-data/gaze/Read directory")
    parser.add_argument("--translate_dir", required=True,
                        help="Path to EMMT preprocessed-data/gaze/Translate directory")
    parser.add_argument("--sentences",     required=True,
                        help="Path to Sentences.csv from the EMMT corpus")
    parser.add_argument("--output",        required=True,
                        help="Output CSV path")
    parser.add_argument(
        "--diagnostics_output",
        help="Optional trial-level line-estimation diagnostics CSV path",
    )
    args = parser.parse_args()

    for d in (args.read_dir, args.translate_dir):
        if not os.path.isdir(d):
            print(f"Directory not found: {d}")
            return

    print("Loading sentences...")
    sentences = load_sentences(args.sentences)
    print(f"  {len(sentences)} sentences loaded.")

    results, diagnostics, read_line_models = [], [], {}
    print("Processing Read stage...")
    process_directory(
        args.read_dir,
        "read",
        sentences,
        results,
        diagnostics,
        read_line_models,
    )
    print("Processing Translate stage...")
    process_directory(
        args.translate_dir,
        "translate",
        sentences,
        results,
        diagnostics,
        read_line_models,
    )

    print("Removing outliers (>2.5 SD from participant mean per stage)...")
    results = remove_outliers(results)

    fields = ["participant", "order", "sentence_id", "ambiguity", "congruency",
              "stage", "word_index", "word", "line_fit_source",
              "total_fixation_duration_ms"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    diagnostics_output = args.diagnostics_output or (
        os.path.splitext(args.output)[0] + "_line_diagnostics.csv"
    )
    Path(diagnostics_output).parent.mkdir(parents=True, exist_ok=True)
    diagnostic_fields = [
        "participant", "order", "sentence_id", "ambiguity", "congruency",
        "stage", "file", "status", "reason", "fit_source", "line_y",
        "alpha", "beta", "half_width_px", "residual_mad_px",
        "n_raw_bouts", "n_candidate_bouts", "n_line_bouts", "n_x_bins",
        "n_word_aois", "candidate_duration_ms", "line_duration_ms",
        "line_bout_share", "line_duration_share", "line_x_span_px",
        "robust_x_span_norm", "mode_score_ratio",
        "independent_failure_reason", "read_alpha", "read_beta",
        "translation_read_shift_px", "n_mapped_word_bouts",
        "n_offtext_bouts", "n_unknown_bouts", "mapped_word_duration_ms",
        "offtext_duration_ms", "unknown_duration_ms", "quality_flags",
        "review_flags",
    ]
    with open(diagnostics_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=diagnostic_fields)
        writer.writeheader()
        writer.writerows(diagnostics)

    read_n  = sum(1 for r in results if r["stage"] == "read")
    trans_n = sum(1 for r in results if r["stage"] == "translate")
    print(f"\nDone.  READ: {read_n} rows,  TRANSLATE: {trans_n} rows")
    print(f"Output: {args.output}")
    accepted = sum(d["status"] == "ok" for d in diagnostics)
    print(
        f"Line estimates: {accepted}/{len(diagnostics)} trials accepted; "
        f"diagnostics: {diagnostics_output}"
    )


if __name__ == "__main__":
    main()
