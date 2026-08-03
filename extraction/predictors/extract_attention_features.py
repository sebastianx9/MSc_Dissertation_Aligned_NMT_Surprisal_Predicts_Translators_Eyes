"""Extract Lim et al. (2024) normalized source-side attention features.

The feature definitions follow ``src_seq_att`` in
ZhengWeiLim/pred-trans-difficulty-NMT at commit ``2265d5c`` and are applied to
the Marian model's own four-beam translation. A source word is treated as one
segment containing all of its Marian subword positions. Context excludes the
whole segment, encoder entropy is computed over all non-special source
positions (the word plus its context), and every feature is divided by the
value obtained from uniform attention over the same positions.

The six source-side features defined by Lim et al. are retained:

* ``attn_entropy``: encoder-attention entropy from the word to the source;
* ``attn_context``: encoder flow from the word to its context;
* ``attn_self``: encoder flow from the word to itself;
* ``attn_eos``: encoder flow from the word to EOS;
* ``attn_recv``: encoder flow received by the word from its context; and
* ``attn_cross``: decoder-to-word cross-attention flow.

Layers and heads are averaged after the segment-level feature and its uniform
dummy value have been calculated, matching Lim et al.'s released code.
"""

import argparse
import csv
import math

import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer


DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-cs"
DEFAULT_REVISION = "2820c6a540ddc2b7c4ea4c95c39b3150bd3ac27e"


def load_sentences(path):
    sentences = {}
    with open(path, newline="", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            sentence_id, text = line.split(",", 1)
            sentences[sentence_id.strip()] = text.strip()
    return sentences


def subwords_to_word_map(tokens):
    """Map Marian/SentencePiece tokens to whitespace-delimited source words."""
    word_map = []
    word_index = -1
    for token in tokens:
        if token in ("<pad>", "</s>"):
            word_map.append(-1)
            continue
        if token.startswith("▁") or word_index == -1:
            word_index += 1
        word_map.append(word_index)
    return word_map


def _safe_ratio(value, dummy, label):
    if not np.isfinite(dummy) or dummy <= 0:
        raise ValueError(f"Uniform-attention dummy for {label} is {dummy}.")
    result = float(value / dummy)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite normalized value for {label}.")
    return result


def lim_normalized_word_features(
    encoder_attention,
    cross_attention,
    word_map,
    eos_index,
    valid_source_positions=None,
):
    """Compute Lim-style normalized features from stacked attention arrays.

    Parameters
    ----------
    encoder_attention
        Array with shape ``(layers, heads, source_queries, source_keys)``.
    cross_attention
        Array with shape ``(layers, heads, target_queries, source_keys)``.
    word_map
        Source-position to word-index mapping; special positions are ``-1``.
    eos_index
        Source position of the EOS token.
    valid_source_positions
        Non-padding source positions.  Uniform dummy attention is defined over
        these positions, just as in Lim et al.'s ``dummy_attention_by_batch``.
    """
    encoder_attention = np.asarray(encoder_attention, dtype=np.float64)
    cross_attention = np.asarray(cross_attention, dtype=np.float64)
    word_map = np.asarray(word_map, dtype=np.int64)

    if encoder_attention.ndim != 4 or cross_attention.ndim != 4:
        raise ValueError("Expected four-dimensional encoder and cross attention.")
    if encoder_attention.shape[2] != encoder_attention.shape[3]:
        raise ValueError("Encoder attention must be square in source positions.")
    source_length = encoder_attention.shape[3]
    if cross_attention.shape[3] != source_length or len(word_map) != source_length:
        raise ValueError("Attention arrays and word map disagree on source length.")
    if not 0 <= eos_index < source_length:
        raise ValueError("EOS index is outside the source sequence.")

    if valid_source_positions is None:
        valid_source_positions = np.arange(source_length, dtype=np.int64)
    else:
        valid_source_positions = np.asarray(
            valid_source_positions, dtype=np.int64
        )
    if len(valid_source_positions) == 0:
        raise ValueError("No valid source positions were supplied.")
    if eos_index not in set(valid_source_positions.tolist()):
        raise ValueError("EOS must be a valid, non-padding source position.")

    lexical_positions = np.flatnonzero(word_map >= 0)
    if len(lexical_positions) < 2:
        raise ValueError("At least two lexical source positions are required.")
    n_words = int(word_map.max()) + 1
    if set(word_map[lexical_positions].tolist()) != set(range(n_words)):
        raise ValueError("Word indices must be contiguous from zero.")

    uniform_denominator = float(len(valid_source_positions))
    n_target_queries = cross_attention.shape[2]
    lexical_entropy = math.log(len(lexical_positions))
    rows = []

    for word_index in range(n_words):
        segment = np.flatnonzero(word_map == word_index)
        context = lexical_positions[word_map[lexical_positions] != word_index]
        if len(segment) == 0 or len(context) == 0:
            raise ValueError(f"Word {word_index} has no segment or context tokens.")

        # Lim et al.'s H(A_e, u, x): for each query subword in u, renormalize
        # over all lexical source positions (u plus its context), sum across
        # the segment, then average across layers and heads.
        lexical_rows = encoder_attention[:, :, segment, :][:, :, :, lexical_positions]
        row_totals = lexical_rows.sum(axis=-1, keepdims=True)
        probabilities = np.divide(
            lexical_rows,
            row_totals,
            out=np.zeros_like(lexical_rows),
            where=row_totals > 0,
        )
        entropy_terms = np.zeros_like(probabilities)
        positive = probabilities > 0
        entropy_terms[positive] = (
            -probabilities[positive] * np.log(probabilities[positive])
        )
        entropy_raw = entropy_terms.sum(axis=(-2, -1)).mean()

        word_to_context_raw = (
            encoder_attention[:, :, segment, :][:, :, :, context]
            .sum(axis=(-2, -1))
            .mean()
        )
        word_to_self_raw = (
            encoder_attention[:, :, segment, :][:, :, :, segment]
            .sum(axis=(-2, -1))
            .mean()
        )
        context_to_word_raw = (
            encoder_attention[:, :, context, :][:, :, :, segment]
            .sum(axis=(-2, -1))
            .mean()
        )
        word_to_eos_raw = (
            encoder_attention[:, :, segment, eos_index].sum(axis=-1).mean()
        )
        target_to_word_raw = (
            cross_attention[:, :, :, segment].sum(axis=(-2, -1)).mean()
        )

        segment_size = float(len(segment))
        context_size = float(len(context))
        dummy_entropy = segment_size * lexical_entropy
        dummy_context = segment_size * context_size / uniform_denominator
        dummy_self = segment_size * segment_size / uniform_denominator
        dummy_eos = segment_size / uniform_denominator
        dummy_cross = (
            float(n_target_queries) * segment_size / uniform_denominator
        )

        rows.append(
            {
                "word_index": word_index,
                "attn_entropy": _safe_ratio(
                    entropy_raw, dummy_entropy, "attn_entropy"
                ),
                "attn_context": _safe_ratio(
                    word_to_context_raw, dummy_context, "attn_context"
                ),
                "attn_self": _safe_ratio(
                    word_to_self_raw, dummy_self, "attn_self"
                ),
                "attn_eos": _safe_ratio(
                    word_to_eos_raw, dummy_eos, "attn_eos"
                ),
                "attn_recv": _safe_ratio(
                    context_to_word_raw, dummy_context, "attn_recv"
                ),
                "attn_cross": _safe_ratio(
                    target_to_word_raw, dummy_cross, "attn_cross"
                ),
            }
        )
    return rows


def extract_features(sentence_text, tokenizer, model, device):
    encoded = tokenizer(
        [sentence_text], return_tensors="pt", truncation=True, max_length=128
    ).to(device)
    source_ids = encoded["input_ids"][0]
    source_tokens = tokenizer.convert_ids_to_tokens(source_ids.tolist())
    word_map = subwords_to_word_map(source_tokens)
    words = sentence_text.split()
    mapped_words = max(word_map) + 1
    if mapped_words != len(words):
        raise ValueError(
            f"Tokenizer produced {mapped_words} word segments for "
            f"{len(words)} whitespace words: {sentence_text!r}"
        )

    eos_positions = (
        source_ids == tokenizer.eos_token_id
    ).nonzero(as_tuple=False).view(-1)
    if len(eos_positions) != 1:
        raise ValueError(f"Expected one source EOS token, found {len(eos_positions)}.")
    eos_index = int(eos_positions.item())

    with torch.no_grad():
        generated_ids = model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            num_beams=4,
            max_length=200,
        )[0]
    if len(generated_ids) < 2:
        raise ValueError("Generated sequence is too short for decoder attention.")
    decoder_input_ids = generated_ids[:-1].unsqueeze(0)

    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )
    if output.encoder_attentions is None or output.cross_attentions is None:
        raise RuntimeError("The model did not return encoder and cross attention.")

    encoder_attention = torch.stack(output.encoder_attentions, dim=0)[
        :, 0
    ].detach().cpu().numpy()
    cross_attention = torch.stack(output.cross_attentions, dim=0)[
        :, 0
    ].detach().cpu().numpy()
    valid_source_positions = (
        encoded["attention_mask"][0].nonzero(as_tuple=False).view(-1).cpu().numpy()
    )
    feature_rows = lim_normalized_word_features(
        encoder_attention=encoder_attention,
        cross_attention=cross_attention,
        word_map=word_map,
        eos_index=eos_index,
        valid_source_positions=valid_source_positions,
    )
    for row, word in zip(feature_rows, words):
        row["word"] = word
        for feature in (
            "attn_entropy",
            "attn_context",
            "attn_self",
            "attn_eos",
            "attn_recv",
            "attn_cross",
        ):
            row[feature] = round(row[feature], 6)
    return feature_rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract Lim-style normalized source attention features."
    )
    parser.add_argument(
        "--sentences", required=True, help="Path to EMMT Sentences.csv"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Hugging Face model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION,
        help="Pinned Hugging Face model revision",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} at revision {args.revision} on {device} ...")
    tokenizer = MarianTokenizer.from_pretrained(
        args.model, revision=args.revision
    )
    model = MarianMTModel.from_pretrained(
        args.model, revision=args.revision, attn_implementation="eager"
    )
    model.eval()
    model.to(device)

    sentences = load_sentences(args.sentences)
    print(f"Loaded {len(sentences)} sentences.\n")
    all_rows = []
    for sentence_id in sorted(sentences):
        text = sentences[sentence_id]
        print(f"  {sentence_id}: {text[:65]}")
        rows = extract_features(text, tokenizer, model, device)
        for row in rows:
            row["sentence_id"] = sentence_id
        all_rows.extend(rows)

    expected_rows = sum(len(text.split()) for text in sentences.values())
    keys = [(row["sentence_id"], row["word_index"]) for row in all_rows]
    if len(all_rows) != expected_rows or len(set(keys)) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} unique word rows, produced {len(all_rows)}."
        )

    fields = [
        "sentence_id", "word_index", "word", "attn_entropy",
        "attn_context", "attn_self", "attn_eos", "attn_recv",
        "attn_cross",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nDone. {len(all_rows)} word rows -> {args.output}")


if __name__ == "__main__":
    main()
