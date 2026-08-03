"""
Extract word-level eye movement measures for READ and TRANSLATE stages:
  - TFD  : total fixation duration (ms)
  - FFD  : first fixation duration (ms)  — first bout on word
  - GD   : first-pass gaze duration (ms) — consecutive bouts from first landing
             until eye leaves word region (in any direction)
  - Go-past: go-past time (ms)          — fixation time from first landing
             until, but excluding, the first subsequent fixation to the right;
             fixations on the word and on prior words are included
  - RRT  : re-reading time (ms)          — all fixations on word after first pass
  - n_fix: total number of fixation bouts on word
  - reread_occurrence: 1 if word is re-fixated after its first visit, else 0
  - regress_in: backward-compatible alias for reread_occurrence; it does not
                imply that the return necessarily came from the word's right
  - first_encounter_status: whether the first visit was progressive or only
                occurred after a word to the right had already been visited

Two-stage theoretical motivation:
  FFD / GD / go-past capture first-pass processing (go-past also includes
  regressions to prior text before the first rightward crossing)
  RRT / reread_occurrence capture re-consultation during translation production
  (attention features may have signal here)

Same word x-position mapping as extract_fixation_duration.py.
Output: one row per participant x sentence x stage x word (fixated words only).
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
    return map_x_to_word_index(x, ranges)


def extract_bouts(
    filepath,
    words,
    stage="read",
    prior_model=None,
    return_diagnostic=False,
    return_model=False,
):
    """
    Return an ordered list of (word_index, duration_ms) after estimating the
    trial's dominant text-line fixation band. Bouts outside the sentence ROI
    are retained as word index -1 so that they terminate first-pass sequences.
    """
    ranges = word_x_ranges(words)
    all_bouts, line_bouts, diagnostic, model = extract_trial_bouts(
        filepath, ranges, stage=stage, prior_model=prior_model
    )
    mapped_bouts = map_trial_bouts(
        all_bouts, line_bouts, ranges, diagnostic=diagnostic
    )
    if not line_bouts:
        result = []
        if return_diagnostic and return_model:
            return result, diagnostic, model
        return (result, diagnostic) if return_diagnostic else result

    if return_diagnostic and return_model:
        return mapped_bouts, diagnostic, model
    if return_diagnostic:
        return mapped_bouts, diagnostic
    return mapped_bouts


def compute_measures(bouts, n_words):
    """
    Given ordered bout list, compute per-word eye movement measures.
    Returns {word_index: dict_of_measures} for fixated words only.
    """
    tfd   = [0.0]  * n_words
    ffd   = [None] * n_words
    gd    = [0.0]  * n_words
    go_past = [None] * n_words
    go_past_status = ["unfixated"] * n_words
    first_encounter_status = ["unfixated"] * n_words
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

    # Pass 2: GD (consecutive bouts from first landing), go-past (regression
    # path), and RRT (remainder). Go-past follows Lijewska et al.'s definition:
    # starting with a word's first fixation, sum fixation durations on that
    # word and on prior words until (but excluding) the first later fixation
    # to its right.
    # It is structurally undefined if that crossing never occurs. A word first
    # encountered only after a word to its right has already been fixated is
    # also left undefined: that landing is a regression-in, not a genuine
    # first-pass encounter with the word.
    for w in range(n_words):
        idx = first_fix_bout[w]
        if idx is None:
            continue
        # accumulate GD: consecutive bouts on w starting at first landing
        i = idx
        while i < len(bouts) and bouts[i][0] == w:
            gd[w] += bouts[i][1]
            i += 1

        previously_visited_right = any(
            0 <= prior_wi < n_words and prior_wi > w
            for prior_wi, _ in bouts[:idx]
        )
        if previously_visited_right:
            first_encounter_status[w] = "regression"
            go_past_status[w] = "first_encounter_by_regression"
        else:
            first_encounter_status[w] = "progressive"
            crossing_idx = next(
                (
                    j
                    for j in range(idx + 1, len(bouts))
                    if 0 <= bouts[j][0] < n_words and bouts[j][0] > w
                ),
                None,
            )
            if crossing_idx is None:
                go_past_status[w] = "no_rightward_crossing"
            elif any(wi < 0 for wi, _ in bouts[idx + 1:crossing_idx]):
                # Once gaze leaves the mapped sentence region, the regression
                # path cannot be reconstructed without silently deleting time.
                go_past_status[w] = "intervening_unmapped_bout"
            else:
                go_past[w] = sum(
                    dur
                    for wi, dur in bouts[idx:crossing_idx]
                    if 0 <= wi < n_words
                )
                go_past_status[w] = "observed"

        # RRT: any later fixations on w
        for j in range(i, len(bouts)):
            if bouts[j][0] == w:
                rrt[w] += bouts[j][1]

    return {
        w: {
            "tfd_ms":      round(tfd[w], 2),
            "ffd_ms":      round(ffd[w], 2),
            "gd_ms":       round(gd[w], 2),
            "go_past_ms":  (
                round(go_past[w], 2) if go_past[w] is not None else None
            ),
            "go_past_status": go_past_status[w],
            "first_encounter_status": first_encounter_status[w],
            "rrt_ms":      round(rrt[w], 2),
            "n_fix":       n_fix[w],
            "reread_occurrence": int(rrt[w] > 0),
            # Backward-compatible alias. This is a return to the word after
            # its first visit, not necessarily a regression from its right.
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
        fpath  = os.path.join(directory, fname)
        trial_key = (participant, order, sentence_id)
        prior_model = (
            read_line_models.get(trial_key)
            if stage_label == "translate"
            else None
        )
        bouts, diagnostic, model = extract_bouts(
            fpath,
            words,
            stage=stage_label,
            prior_model=prior_model,
            return_diagnostic=True,
            return_model=True,
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
                "line_fit_source": model.fit_source,
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
    parser = argparse.ArgumentParser(
        description="Extract word-level eye-movement measures from EMMT gaze files."
    )
    parser.add_argument("--read_dir", required=True,
                        help="Path to EMMT preprocessed-data/gaze/Read directory")
    parser.add_argument("--translate_dir", required=True,
                        help="Path to EMMT preprocessed-data/gaze/Translate directory")
    parser.add_argument("--sentences", required=True,
                        help="Path to Sentences.csv from the EMMT corpus")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--diagnostics_output",
        help="Optional trial-level line-estimation diagnostics CSV path",
    )
    args = parser.parse_args()

    print("Loading sentences...")
    sentences = load_sentences(args.sentences)
    print(f"  {len(sentences)} sentences.")

    results, diagnostics, read_line_models = [], [], {}
    print("Processing Read...")
    process_directory(
        args.read_dir,
        "read",
        sentences,
        results,
        diagnostics,
        read_line_models,
    )
    print(f"  {sum(1 for r in results if r['stage']=='read')} rows")
    print("Processing Translate...")
    process_directory(
        args.translate_dir,
        "translate",
        sentences,
        results,
        diagnostics,
        read_line_models,
    )
    print(f"  {sum(1 for r in results if r['stage']=='translate')} rows")

    print("Removing outliers (2.5 SD on TFD)...")
    results = remove_outliers(results)

    fields = ["participant", "order", "sentence_id", "ambiguity", "congruency",
              "stage", "word_index", "word",
              "line_fit_source", "tfd_ms", "ffd_ms", "gd_ms", "go_past_ms",
              "go_past_status", "first_encounter_status", "rrt_ms",
              "reread_occurrence",
              "n_fix", "regress_in"]

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
    print(f"\nDone.  Read: {read_n}  Translate: {trans_n}")
    print(f"Output: {args.output}")
    accepted = sum(d["status"] == "ok" for d in diagnostics)
    print(
        f"Line estimates: {accepted}/{len(diagnostics)} trials accepted; "
        f"diagnostics: {diagnostics_output}"
    )


if __name__ == "__main__":
    main()
