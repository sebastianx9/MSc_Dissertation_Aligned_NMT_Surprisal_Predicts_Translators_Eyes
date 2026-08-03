"""
Extract word-level total fixation duration for READ and TRANSLATE stages
from EMMT preprocessed gaze CSV files.

Word x-positions computed from actual font metrics (Arial Bold as
metric-compatible substitute for Free Sans Bold, fontsize 28).

Output: one row per participant x sentence x stage x word
"""

import argparse
import os
import csv
from PIL import ImageFont

# Display parameters from experiment script
TEXT_CENTER_X = 620
TEXT_Y        = 200
Y_TOLERANCE   = 60   # px: fixation must be within this range of TEXT_Y
SPACE_WIDTH   = 8.0  # measured from Arial Bold 28px

# Load font for accurate word width measurement
_font_candidates = [
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
_font_path = next((p for p in _font_candidates if os.path.exists(p)), None)
if _font_path is None:
    raise FileNotFoundError("No suitable font found for word width measurement.")
FONT = ImageFont.truetype(_font_path, 28)


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
    Falls back to nearest word if x is between words.
    Returns -1 if no words defined.
    """
    if not ranges:
        return -1
    # exact match
    for i, (x0, x1, _) in enumerate(ranges):
        if x0 <= x <= x1:
            return i
    # nearest word center
    centers = [(x0 + x1) / 2 for x0, x1, _ in ranges]
    return min(range(len(centers)), key=lambda i: abs(centers[i] - x))


def ts_to_seconds(ts_str):
    """Convert 'HH:MM:SS.SSSS' to float seconds."""
    s = ts_str.strip()
    h, m, rest = s[0:2], s[3:5], s[6:]
    sec, frac = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(frac) / (10 ** len(frac))


def extract_word_fixations(filepath, words):
    """
    Return {word_index: total_fixation_ms} for one gaze CSV file.
    Groups consecutive fixation samples into bouts, maps each bout
    to a word by mean x-position (filtered by y proximity to text line).
    """
    ranges = word_x_ranges(words)
    timestamps, xs, ys, events = [], [], [], []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = ts_to_seconds(row["TimeStamp"])
                ev = row["Event"].strip().lower()
            except (ValueError, KeyError):
                continue
            x_val = row["X"].strip()
            y_val = row["Y"].strip()
            try:
                x = float(x_val) if x_val else None
            except ValueError:
                x = None
            try:
                y = float(y_val) if y_val else None
            except ValueError:
                y = None
            timestamps.append(ts)
            xs.append(x)
            ys.append(y)
            events.append(ev)

    if len(timestamps) < 2:
        return {}

    # median inter-sample interval for bout duration calculation
    diffs = sorted(
        timestamps[i+1] - timestamps[i]
        for i in range(len(timestamps) - 1)
        if timestamps[i+1] > timestamps[i]
    )
    sample_interval = diffs[len(diffs) // 2] if diffs else 0.0005

    word_totals = {}
    in_fix = False
    bout_ts, bout_xs, bout_ys = [], [], []

    def close_bout():
        if not bout_ts:
            return
        mean_y = sum(v for v in bout_ys if v is not None)
        valid_y = [v for v in bout_ys if v is not None]
        if not valid_y:
            return
        mean_y = sum(valid_y) / len(valid_y)
        if abs(mean_y - TEXT_Y) > Y_TOLERANCE:
            return  # fixation not on text line
        valid_x = [v for v in bout_xs if v is not None]
        if not valid_x:
            return
        mean_x = sum(valid_x) / len(valid_x)
        duration_ms = ((bout_ts[-1] - bout_ts[0]) + sample_interval) * 1000
        wi = x_to_word_index(mean_x, ranges)
        if wi >= 0:
            word_totals[wi] = word_totals.get(wi, 0.0) + duration_ms

    for ts, x, y, ev in zip(timestamps, xs, ys, events):
        if ev == "fixation":
            if not in_fix:
                in_fix = True
                bout_ts, bout_xs, bout_ys = [], [], []
            bout_ts.append(ts)
            bout_xs.append(x)
            bout_ys.append(y)
        else:
            if in_fix:
                close_bout()
                in_fix = False

    if in_fix:
        close_bout()

    return word_totals


def parse_filename(fname):
    if not fname.endswith(".csv"):
        return None
    parts = fname[:-4].split("-")
    if len(parts) != 5:
        return None
    return parts  # [participant, order, sentence_id, ambiguity, congruency]


def process_directory(directory, stage_label, sentences, results):
    for fname in sorted(os.listdir(directory)):
        parts = parse_filename(fname)
        if parts is None:
            continue
        participant, order, sentence_id, ambiguity, congruency = parts
        words = sentences.get(sentence_id)
        if words is None:
            continue
        fpath = os.path.join(directory, fname)
        word_totals = extract_word_fixations(fpath, words)
        for wi, dur in sorted(word_totals.items()):
            if dur < 20:
                continue
            results.append({
                "participant":               participant,
                "order":                     order,
                "sentence_id":              sentence_id,
                "ambiguity":                ambiguity,
                "congruency":               congruency,
                "stage":                    stage_label,
                "word_index":               wi,
                "word":                     words[wi],
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
    args = parser.parse_args()

    for d in (args.read_dir, args.translate_dir):
        if not os.path.isdir(d):
            print(f"Directory not found: {d}")
            return

    print("Loading sentences...")
    sentences = load_sentences(args.sentences)
    print(f"  {len(sentences)} sentences loaded.")

    results = []
    print("Processing Read stage...")
    process_directory(args.read_dir,      "read",      sentences, results)
    print("Processing Translate stage...")
    process_directory(args.translate_dir, "translate", sentences, results)

    print("Removing outliers (>2.5 SD from participant mean per stage)...")
    results = remove_outliers(results)

    fields = ["participant", "order", "sentence_id", "ambiguity", "congruency",
              "stage", "word_index", "word", "total_fixation_duration_ms"]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    read_n  = sum(1 for r in results if r["stage"] == "read")
    trans_n = sum(1 for r in results if r["stage"] == "translate")
    print(f"\nDone.  READ: {read_n} rows,  TRANSLATE: {trans_n} rows")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
