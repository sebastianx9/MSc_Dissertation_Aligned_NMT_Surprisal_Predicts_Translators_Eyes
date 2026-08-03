"""
Extract word-level eye movement measures for READ and TRANSLATE stages:
  - TFD  : total fixation duration (ms)
  - FFD  : first fixation duration (ms)  — first bout on word
  - GD   : first-pass gaze duration (ms) — consecutive bouts from first landing
             until eye leaves word region (in any direction)
  - RRT  : re-reading time (ms)          — all fixations on word after first pass
  - n_fix: total number of fixation bouts on word
  - regress_in: 1 if word re-fixated after first pass, else 0

Two-stage theoretical motivation:
  FFD / GD capture initial lexical access (surprisal expected here)
  RRT / regress_in capture re-consultation during translation production
  (attention features may have signal here)

Same word x-position mapping as extract_fixation_duration.py.
Output: one row per participant x sentence x stage x word (fixated words only).
"""

import os
import csv
from PIL import ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────
READ_DIR      = "/Users/sebastianx/eyetracked-multi-modal-translation/preprocessed-data/gaze/Read"
TRANSLATE_DIR = "/Users/sebastianx/eyetracked-multi-modal-translation/preprocessed-data/gaze/Translate"
SENTENCES_CSV = "/Users/sebastianx/eyetracked-multi-modal-translation/probes/Sentences.csv"
OUTPUT_CSV    = "/Users/sebastianx/Dissertation_Data/eye_measures_word.csv"
# ───────────────────────────────────────────────────────────────────────────

TEXT_CENTER_X = 620
TEXT_Y        = 200
Y_TOLERANCE   = 60
SPACE_WIDTH   = 8.0
MIN_FIX_MS    = 20    # discard bouts shorter than this

_font_candidates = [
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
_font_path = next((p for p in _font_candidates if os.path.exists(p)), None)
if _font_path is None:
    raise FileNotFoundError("No suitable font found.")
FONT = ImageFont.truetype(_font_path, 28)


def word_pixel_width(word):
    bbox = FONT.getbbox(word)
    return bbox[2] - bbox[0]


def load_sentences(path):
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
    widths = [word_pixel_width(w) for w in words]
    total_width = sum(widths) + SPACE_WIDTH * (len(words) - 1)
    x = TEXT_CENTER_X - total_width / 2
    ranges = []
    for w, ww in zip(words, widths):
        ranges.append((x, x + ww, w))
        x += ww + SPACE_WIDTH
    return ranges


def x_to_word_index(x, ranges):
    if not ranges:
        return -1
    for i, (x0, x1, _) in enumerate(ranges):
        if x0 <= x <= x1:
            return i
    centers = [(x0 + x1) / 2 for x0, x1, _ in ranges]
    return min(range(len(centers)), key=lambda i: abs(centers[i] - x))


def ts_to_seconds(ts_str):
    s = ts_str.strip()
    h, m, rest = s[0:2], s[3:5], s[6:]
    sec, frac = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(frac) / (10 ** len(frac))


def extract_bouts(filepath, words):
    """
    Return ordered list of (word_index, duration_ms) for all valid fixation
    bouts on the text line, in temporal order.
    """
    ranges = word_x_ranges(words)
    timestamps, xs, ys, events = [], [], [], []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts  = ts_to_seconds(row["TimeStamp"])
                ev  = row["Event"].strip().lower()
            except (ValueError, KeyError):
                continue
            try:
                x = float(row["X"].strip()) if row["X"].strip() else None
            except ValueError:
                x = None
            try:
                y = float(row["Y"].strip()) if row["Y"].strip() else None
            except ValueError:
                y = None
            timestamps.append(ts)
            xs.append(x)
            ys.append(y)
            events.append(ev)

    if len(timestamps) < 2:
        return []

    diffs = sorted(
        timestamps[i+1] - timestamps[i]
        for i in range(len(timestamps) - 1)
        if timestamps[i+1] > timestamps[i]
    )
    sample_interval = diffs[len(diffs) // 2] if diffs else 0.0005

    bouts = []
    in_fix = False
    bout_ts, bout_xs, bout_ys = [], [], []

    def close_bout():
        if not bout_ts:
            return
        valid_y = [v for v in bout_ys if v is not None]
        if not valid_y:
            return
        if abs(sum(valid_y) / len(valid_y) - TEXT_Y) > Y_TOLERANCE:
            return
        valid_x = [v for v in bout_xs if v is not None]
        if not valid_x:
            return
        mean_x    = sum(valid_x) / len(valid_x)
        dur_ms    = ((bout_ts[-1] - bout_ts[0]) + sample_interval) * 1000
        if dur_ms < MIN_FIX_MS:
            return
        wi = x_to_word_index(mean_x, ranges)
        if wi >= 0:
            bouts.append((wi, dur_ms))

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

    return bouts


def compute_measures(bouts, n_words):
    """
    Given ordered bout list, compute per-word eye movement measures.
    Returns {word_index: dict_of_measures} for fixated words only.
    """
    tfd   = [0.0]  * n_words
    ffd   = [None] * n_words
    gd    = [0.0]  * n_words
    rrt   = [0.0]  * n_words
    n_fix = [0]    * n_words
    first_fix_bout = [None] * n_words  # index of first bout on word

    # Pass 1: TFD, FFD, n_fix, first-bout index
    for i, (wi, dur) in enumerate(bouts):
        if wi < 0 or wi >= n_words:
            continue
        tfd[wi]   += dur
        n_fix[wi] += 1
        if first_fix_bout[wi] is None:
            first_fix_bout[wi] = i
            ffd[wi] = dur

    # Pass 2: GD (consecutive bouts from first landing) and RRT (remainder)
    for w in range(n_words):
        idx = first_fix_bout[w]
        if idx is None:
            continue
        # accumulate GD: consecutive bouts on w starting at first landing
        i = idx
        while i < len(bouts) and bouts[i][0] == w:
            gd[w] += bouts[i][1]
            i += 1
        # RRT: any later fixations on w
        for j in range(i, len(bouts)):
            if bouts[j][0] == w:
                rrt[w] += bouts[j][1]

    return {
        w: {
            "tfd_ms":      round(tfd[w], 2),
            "ffd_ms":      round(ffd[w], 2),
            "gd_ms":       round(gd[w], 2),
            "rrt_ms":      round(rrt[w], 2),
            "n_fix":       n_fix[w],
            "regress_in":  int(rrt[w] > 0),
        }
        for w in range(n_words)
        if ffd[w] is not None
    }


def parse_filename(fname):
    if not fname.endswith(".csv"):
        return None
    parts = fname[:-4].split("-")
    if len(parts) != 5:
        return None
    return parts


def process_directory(directory, stage_label, sentences, results):
    for fname in sorted(os.listdir(directory)):
        parts = parse_filename(fname)
        if parts is None:
            continue
        participant, order, sentence_id, ambiguity, congruency = parts
        words = sentences.get(sentence_id)
        if words is None:
            continue
        fpath  = os.path.join(directory, fname)
        bouts  = extract_bouts(fpath, words)
        if not bouts:
            continue
        measures = compute_measures(bouts, len(words))
        for wi, m in sorted(measures.items()):
            results.append({
                "participant":  participant,
                "order":        order,
                "sentence_id":  sentence_id,
                "ambiguity":    ambiguity,
                "congruency":   congruency,
                "stage":        stage_label,
                "word_index":   wi,
                "word":         words[wi],
                **m,
            })


def remove_outliers(results, sd_threshold=2.5):
    from collections import defaultdict
    import math
    groups = defaultdict(list)
    for r in results:
        groups[(r["participant"], r["stage"])].append(r["tfd_ms"])
    stats = {}
    for key, vals in groups.items():
        mean = sum(vals) / len(vals)
        sd   = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
        stats[key] = (mean, sd)
    before = len(results)
    filtered = [
        r for r in results
        if abs(r["tfd_ms"] - stats[(r["participant"], r["stage"])][0])
           <= sd_threshold * stats[(r["participant"], r["stage"])][1]
    ]
    print(f"  Outliers removed: {before - len(filtered)} / {before}")
    return filtered


def main():
    print("Loading sentences...")
    sentences = load_sentences(SENTENCES_CSV)
    print(f"  {len(sentences)} sentences.")

    results = []
    print("Processing Read...")
    process_directory(READ_DIR,      "read",      sentences, results)
    print(f"  {sum(1 for r in results if r['stage']=='read')} rows")
    print("Processing Translate...")
    process_directory(TRANSLATE_DIR, "translate", sentences, results)
    print(f"  {sum(1 for r in results if r['stage']=='translate')} rows")

    print("Removing outliers (2.5 SD on TFD)...")
    results = remove_outliers(results)

    fields = ["participant", "order", "sentence_id", "ambiguity", "congruency",
              "stage", "word_index", "word",
              "tfd_ms", "ffd_ms", "gd_ms", "rrt_ms", "n_fix", "regress_in"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    read_n  = sum(1 for r in results if r["stage"] == "read")
    trans_n = sum(1 for r in results if r["stage"] == "translate")
    print(f"\nDone.  Read: {read_n}  Translate: {trans_n}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
