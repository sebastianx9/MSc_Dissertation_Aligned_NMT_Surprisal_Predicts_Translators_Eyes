#!/usr/bin/env python3
"""Extract source-aligned NMT surprisal and effective alignment mass.

For target token t and source word w, a_t,w is final-layer cross-attention
averaged over heads, summed over the source subwords of w, and normalised over
source words. The output contains

    alignment_mass(w) = sum_t a_t,w
    surprisal_soft(w) = sum_t surprisal(t) * a_t,w
    surprisal_per_mass(w) = surprisal_soft(w) / alignment_mass(w)

The recomputed surprisal column can be compared exactly with the primary
``nmt_surprisal_soft_word.csv`` before alignment mass is used in analysis.
"""

import argparse
import csv

import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer


DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-cs"
DEFAULT_REVISION = "2820c6a540ddc2b7c4ea4c95c39b3150bd3ac27e"
SPECIAL_TOKENS = {"<pad>", "</s>", "<unk>"}


def load_sentences(path):
    sentences = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sentence_id, text = line.split(",", 1)
            sentences[sentence_id.strip()] = text.strip()
    return sentences


def subwords_to_word_map(tokens):
    mapping = []
    word_index = -1
    for token in tokens:
        if token in SPECIAL_TOKENS:
            mapping.append(-1)
            continue
        if token.startswith("▁") or word_index == -1:
            word_index += 1
        mapping.append(word_index)
    return mapping


def process_sentence(text, tokenizer, model, device):
    encoded = tokenizer(
        [text], return_tensors="pt", truncation=True, max_length=128
    ).to(device)
    source_tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][0].tolist()
    )

    with torch.no_grad():
        generated_ids = model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            num_beams=4,
            max_length=200,
        )[0]

    decoder_input_ids = generated_ids[:-1].unsqueeze(0)
    target_ids = generated_ids[1:].unsqueeze(0)
    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
        )

    log_probs = torch.nn.functional.log_softmax(output.logits[0], dim=-1)
    token_surprisals = np.asarray(
        [
            -log_probs[index, token_id].item()
            for index, token_id in enumerate(target_ids[0].tolist())
        ]
    )
    attention = output.cross_attentions[-1][0].mean(dim=0).cpu().numpy()

    target_tokens = tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
    source_words = text.split()
    source_word_map = subwords_to_word_map(source_tokens)
    word_attention = np.zeros((len(target_tokens), len(source_words)))
    for subword_index, word_index in enumerate(source_word_map):
        if 0 <= word_index < len(source_words) and subword_index < attention.shape[1]:
            word_attention[:, word_index] += attention[:, subword_index]

    soft_surprisal = np.zeros(len(source_words))
    alignment_mass = np.zeros(len(source_words))
    for target_index, (token, surprisal) in enumerate(
        zip(target_tokens, token_surprisals)
    ):
        if token in SPECIAL_TOKENS:
            continue
        row = word_attention[target_index]
        row_sum = row.sum()
        if row_sum <= 0:
            continue
        normalised_attention = row / row_sum
        alignment_mass += normalised_attention
        soft_surprisal += surprisal * normalised_attention

    rows = []
    for word_index, word in enumerate(source_words):
        mass = float(alignment_mass[word_index])
        soft = float(soft_surprisal[word_index])
        rows.append(
            {
                "word_index": word_index,
                "word": word,
                "surprisal_soft_recomputed": round(soft, 6),
                "alignment_mass": round(mass, 6),
                "surprisal_per_mass": (
                    round(soft / mass, 6) if mass > 0 else None
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract word-level soft-alignment mass."
    )
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Loading {args.model} at revision {args.revision} on {device} ...",
        flush=True,
    )
    tokenizer = MarianTokenizer.from_pretrained(
        args.model, revision=args.revision
    )
    model = MarianMTModel.from_pretrained(
        args.model,
        revision=args.revision,
        output_attentions=True,
    )
    model.eval()
    model.to(device)

    sentences = load_sentences(args.sentences)
    all_rows = []
    for index, sentence_id in enumerate(sorted(sentences), start=1):
        rows = process_sentence(
            sentences[sentence_id], tokenizer, model, device
        )
        for row in rows:
            row["sentence_id"] = sentence_id
        all_rows.extend(rows)
        print(f"[{index:3d}/{len(sentences)}] {sentence_id}", flush=True)

    fields = [
        "sentence_id",
        "word_index",
        "word",
        "surprisal_soft_recomputed",
        "alignment_mass",
        "surprisal_per_mass",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Done: {len(all_rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
