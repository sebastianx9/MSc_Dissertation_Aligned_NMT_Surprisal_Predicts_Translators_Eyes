#!/usr/bin/env python3
"""Describe the distribution of c_nmt and c_mono across source-word POS.

This is a construct-interpretation analysis, not a test of POS moderation of
the surprisal-to-fixation slope. Source sentences are reconstructed from the
word-indexed NMT file so that POS indices match the surprisal extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import spacy


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
STOPLIGHT = ("S003", 3)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spacy-model", default="en_core_web_lg")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-stoplight", action="store_true")
    return parser.parse_args()


def zscore(values: pd.Series) -> pd.Series:
    return (values - values.mean()) / values.std(ddof=1)


def classify(pos: str) -> str:
    if pos in CONTENT_POS:
        return "content"
    if pos in FUNCTION_POS:
        return "function"
    return "other"


def tag_words(nmt: pd.DataFrame, nlp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sentence_id, sentence in nmt.groupby("sentence_id", sort=True):
        sentence = sentence.sort_values("word_index")
        indices = sentence["word_index"].astype(int).tolist()
        if indices != list(range(len(indices))):
            raise ValueError(f"Non-contiguous word indices in {sentence_id}")
        words = sentence["word"].astype(str).tolist()
        text = " ".join(words)
        doc = nlp(text)
        cursor = 0
        for index, word in zip(indices, words):
            start, end = cursor, cursor + len(word)
            cursor = end + 1
            candidates = [
                token for token in doc
                if token.idx < end and token.idx + len(token.text) > start
                and not token.is_space and not token.is_punct
            ]
            if not candidates:
                candidates = [
                    token for token in doc
                    if token.idx < end and token.idx + len(token.text) > start
                ]
            if not candidates:
                raise ValueError(
                    f"No spaCy token aligned to {sentence_id}:{index} {word!r}"
                )
            token = max(
                candidates,
                key=lambda item: min(end, item.idx + len(item.text))
                - max(start, item.idx),
            )
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "word_index": index,
                    "word": word,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "word_class": classify(token.pos_),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_difference(
    data: pd.DataFrame, outcome: str, repetitions: int, seed: int
) -> tuple[float, float, float]:
    used = data[data["word_class"].isin(["content", "function"])].copy()
    observed = (
        used.loc[used["word_class"] == "function", outcome].mean()
        - used.loc[used["word_class"] == "content", outcome].mean()
    )
    groups = {sid: frame for sid, frame in used.groupby("sentence_id")}
    sentence_ids = np.array(list(groups))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        sampled = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        frame = pd.concat([groups[sid] for sid in sampled], ignore_index=True)
        draws[i] = (
            frame.loc[frame["word_class"] == "function", outcome].mean()
            - frame.loc[frame["word_class"] == "content", outcome].mean()
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(observed), float(lower), float(upper)


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nmt = pd.read_csv(args.data_dir / "nmt_surprisal_soft_word.csv")
    mono = pd.read_csv(args.data_dir / "monolingual_surprisal_word.csv")
    nmt["word_index"] = nmt["word_index"].astype(int)
    mono["word_index"] = mono["word_index"].astype(int)

    nlp = spacy.load(args.spacy_model)
    pos = tag_words(nmt, nlp)

    if not args.include_stoplight:
        nmt = nmt[
            ~((nmt["sentence_id"] == STOPLIGHT[0]) &
              (nmt["word_index"] == STOPLIGHT[1]))
        ].copy()
        mono = mono[
            ~((mono["sentence_id"] == STOPLIGHT[0]) &
              (mono["word_index"] == STOPLIGHT[1]))
        ].copy()

    data = (
        nmt[["sentence_id", "word_index", "word", "surprisal_soft"]]
        .rename(columns={"surprisal_soft": "c_nmt"})
        .merge(
            mono[["sentence_id", "word_index", "surprisal_sum"]]
            .rename(columns={"surprisal_sum": "c_mono"}),
            on=["sentence_id", "word_index"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pos.drop(columns="word"),
            on=["sentence_id", "word_index"],
            how="inner",
            validate="one_to_one",
        )
    )
    data["z_c_nmt"] = zscore(data["c_nmt"])
    data["z_c_mono"] = zscore(data["c_mono"])
    data.to_csv(args.output_dir / "rq1_cnmt_pos.csv", index=False)

    summaries: list[pd.DataFrame] = []
    for grouping in ["word_class", "pos"]:
        summary = (
            data.groupby(grouping, observed=True)
            .agg(
                n=("word_index", "size"),
                mean_z_c_nmt=("z_c_nmt", "mean"),
                median_z_c_nmt=("z_c_nmt", "median"),
                mean_z_c_mono=("z_c_mono", "mean"),
                median_z_c_mono=("z_c_mono", "median"),
            )
            .reset_index(names="group")
        )
        summary.insert(0, "grouping", grouping)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.output_dir / "rq1_cnmt_pos_summary.csv", index=False)

    effects = []
    for outcome in ["z_c_nmt", "z_c_mono"]:
        estimate, lower, upper = cluster_bootstrap_difference(
            data, outcome, args.bootstrap, args.seed
        )
        effects.append(
            {
                "outcome": outcome,
                "contrast": "function - content",
                "estimate": estimate,
                "ci_95_low": lower,
                "ci_95_high": upper,
                "bootstrap_unit": "sentence",
                "bootstrap_repetitions": args.bootstrap,
                "seed": args.seed,
            }
        )
    pd.DataFrame(effects).to_csv(
        args.output_dir / "rq1_cnmt_pos_contrasts.csv", index=False
    )

    complete = data.dropna(subset=["z_c_nmt", "z_c_mono", "pos"]).copy()
    complete["nmt_within_pos"] = complete["z_c_nmt"] - complete.groupby("pos")[
        "z_c_nmt"
    ].transform("mean")
    complete["mono_within_pos"] = complete["z_c_mono"] - complete.groupby("pos")[
        "z_c_mono"
    ].transform("mean")
    core = complete[complete["word_class"].isin(["content", "function"])].copy()
    core["nmt_within_class"] = core["z_c_nmt"] - core.groupby("word_class")[
        "z_c_nmt"
    ].transform("mean")
    core["mono_within_class"] = core["z_c_mono"] - core.groupby("word_class")[
        "z_c_mono"
    ].transform("mean")
    correlations = pd.DataFrame(
        [
            {
                "correlation": "overall",
                "r": complete["z_c_nmt"].corr(complete["z_c_mono"]),
                "n": len(complete),
            },
            {
                "correlation": "within_POS_centered",
                "r": complete["nmt_within_pos"].corr(complete["mono_within_pos"]),
                "n": len(complete),
            },
            {
                "correlation": "within_content_function_centered",
                "r": core["nmt_within_class"].corr(core["mono_within_class"]),
                "n": len(core),
            },
        ]
    )
    correlations.to_csv(
        args.output_dir / "rq1_cnmt_mono_correlations.csv", index=False
    )

    print(f"Analysed {len(data)} source-word positions across "
          f"{data['sentence_id'].nunique()} sentences.")
    print(summary[(summary["grouping"] == "word_class")].to_string(index=False))
    print(pd.DataFrame(effects).to_string(index=False))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
