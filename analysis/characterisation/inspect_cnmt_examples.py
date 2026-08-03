#!/usr/bin/env python3
"""Inspect target-side contributions for selected c_nmt/c_mono divergences."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer


DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-cs"
DEFAULT_REVISION = "2820c6a540ddc2b7c4ea4c95c39b3150bd3ac27e"
EXAMPLES = [
    ("S125", 5, "wheelchair"),
    ("S076", 10, "near"),
    ("S087", 11, "trucks"),
    ("S142", 7, "glasses"),
    ("S136", 8, "push"),
    ("S055", 4, "boarding"),
    ("S186", 8, "driving"),
    ("S104", 19, "outside"),
    ("S141", 5, "set"),
    ("S030", 6, "lot"),
    ("S199", 7, "caught"),
    ("S200", 4, "stamp"),
    # The second occurrence of "food" in "other food trucks".  The first
    # occurrence has a different c_mono value and is not in the divergence set.
    ("S087", 10, "food"),
    ("S036", 5, "move"),
    ("S037", 5, "as"),
]


def subwords_to_word_map(tokens):
    mapping = []
    word_index = -1
    for token in tokens:
        if token in ("<pad>", "</s>", "<unk>"):
            mapping.append(-1)
            continue
        if token.startswith("▁") or word_index == -1:
            word_index += 1
        mapping.append(word_index)
    return mapping


def target_words_and_map(tokens):
    words = []
    mapping = []
    word_index = -1
    for token in tokens:
        if token in ("<pad>", "</s>", "<unk>"):
            mapping.append(-1)
            continue
        piece = token.replace("▁", " ") if token.startswith("▁") else token
        if token.startswith("▁") or word_index == -1:
            word_index += 1
            words.append(piece.strip())
        else:
            words[word_index] += piece
        mapping.append(word_index)
    return words, mapping


def inspect_sentence(text, source_word_index, tokenizer, model):
    encoded = tokenizer(
        [text], return_tensors="pt", truncation=True, max_length=128
    )
    source_tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0].tolist()
    )
    with torch.no_grad():
        generated = model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            num_beams=4,
            max_length=200,
        )[0]

    decoder_input_ids = generated[:-1].unsqueeze(0)
    target_ids = generated[1:].unsqueeze(0)
    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
        )

    log_probs = torch.nn.functional.log_softmax(output.logits[0], dim=-1)
    target_id_list = target_ids[0].tolist()
    surprisals = np.array(
        [-log_probs[t, token_id].item() for t, token_id in enumerate(target_id_list)]
    )
    attention = output.cross_attentions[-1][0].mean(dim=0).cpu().numpy()
    target_tokens = tokenizer.convert_ids_to_tokens(target_id_list)

    source_words = text.split()
    source_map = subwords_to_word_map(source_tokens)
    word_attention = np.zeros((len(target_tokens), len(source_words)))
    for subword_index, word_index in enumerate(source_map):
        if 0 <= word_index < len(source_words) and subword_index < attention.shape[1]:
            word_attention[:, word_index] += attention[:, subword_index]

    row_sums = word_attention.sum(axis=1)
    allocations = np.divide(
        word_attention[:, source_word_index],
        row_sums,
        out=np.zeros_like(row_sums),
        where=row_sums > 0,
    )
    special_tokens = {"</s>", "<pad>", "<unk>"}
    allocations[
        np.array([token in special_tokens for token in target_tokens], dtype=bool)
    ] = 0.0
    token_contributions = surprisals * allocations

    target_words, target_map = target_words_and_map(target_tokens)
    word_contributions = np.zeros(len(target_words))
    for token_index, target_word_index in enumerate(target_map):
        if target_word_index >= 0:
            word_contributions[target_word_index] += token_contributions[token_index]
    ranked = sorted(
        zip(target_words, word_contributions),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked = [(word, value) for word, value in ranked if value > 0][:4]

    token_details = []
    for token_index, (token_id, token, target_word_index) in enumerate(
        zip(target_id_list, target_tokens, target_map)
    ):
        if token in special_tokens:
            continue
        token_details.append(
            {
                "target_token_index": token_index,
                "target_token_id": token_id,
                "target_piece": token,
                "target_word": (
                    target_words[target_word_index]
                    if target_word_index >= 0
                    else ""
                ),
                "conditional_surprisal_nats": float(surprisals[token_index]),
                "normalised_attention_to_source": float(allocations[token_index]),
                "contribution_nats": float(token_contributions[token_index]),
            }
        )

    return {
        "source_sentence": text,
        "model_translation": tokenizer.decode(generated, skip_special_tokens=True),
        "top_target_contributors": "; ".join(
            f"{word} ({value:.2f})" for word, value in ranked
        ),
        "reconstructed_c_nmt": float(token_contributions.sum()),
        "token_details": token_details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--token-output",
        type=Path,
        help="Optional CSV path for the per-target-token decomposition.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    sentences = {
        sentence_id: " ".join(
            group.sort_values("word_index")["word"].astype(str)
        )
        for sentence_id, group in data.groupby("sentence_id")
    }
    tokenizer = MarianTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = MarianMTModel.from_pretrained(
        args.model,
        revision=args.revision,
        output_attentions=True,
        local_files_only=args.local_files_only,
    )
    model.eval()

    rows = []
    token_rows = []
    for sentence_id, word_index, word in EXAMPLES:
        source_row = data.loc[
            data["sentence_id"].eq(sentence_id)
            & data["word_index"].eq(word_index)
        ].iloc[0]
        details = inspect_sentence(
            sentences[sentence_id], word_index, tokenizer, model
        )
        token_details = details.pop("token_details")
        token_rows.extend(
            {
                "sentence_id": sentence_id,
                "source_word_index": word_index,
                "source_word": word,
                **token_detail,
            }
            for token_detail in token_details
        )
        rows.append(
            {
                "sentence_id": sentence_id,
                "word_index": word_index,
                "word": word,
                "pos": source_row["pos"],
                "z_c_nmt": source_row["z_c_nmt"],
                "z_c_mono": source_row["z_c_mono"],
                **details,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if args.token_output:
        args.token_output.parent.mkdir(parents=True, exist_ok=True)
        with args.token_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=token_rows[0].keys())
            writer.writeheader()
            writer.writerows(token_rows)
    for row in rows:
        print(
            f"{row['sentence_id']} {row['word']}: "
            f"c_nmt={row['z_c_nmt']:+.2f}, c_mono={row['z_c_mono']:+.2f}\n"
            f"  EN: {row['source_sentence']}\n"
            f"  CS: {row['model_translation']}\n"
            f"  target contributions: {row['top_target_contributors']}\n"
            f"  reconstructed c_nmt={row['reconstructed_c_nmt']:.6f}"
        )


if __name__ == "__main__":
    main()
