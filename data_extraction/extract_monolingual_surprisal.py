"""
Compute word-level monolingual surprisal for English source sentences
using GPT-2 (causal language model).

surprisal(word_t) = sum of -log P(subword_i | all previous tokens)
                    for all subwords in word_t

First word's surprisal = -log P from unconditional GPT-2 distribution
(no prior context), treated as valid but noted in output (n_tokens still set).

Output columns:
  sentence_id, word_index, word,
  surprisal_sum,   # sum of -log P for subwords in this word
  surprisal_mean,  # mean of -log P for subwords in this word
  n_tokens         # number of GPT-2 subword tokens in this word
"""

import argparse
import csv
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


DEFAULT_MODEL = "gpt2"
DEFAULT_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def load_sentences(path):
    sentences = {}
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, text = line.split(",", 1)
            sentences[sid.strip()] = text.strip()
    return sentences


def compute_surprisal(sentence_text, tokenizer, model):
    enc = tokenizer(
        sentence_text,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]

    with torch.no_grad():
        logits = model(input_ids).logits[0]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    with torch.no_grad():
        logits_empty = model(
            torch.tensor([[tokenizer.eos_token_id]])
        ).logits[0, 0]
    lp0 = torch.nn.functional.log_softmax(logits_empty, dim=-1)

    ids = input_ids[0].tolist()
    token_surprisals = [-lp0[ids[0]].item()]
    for t in range(1, len(ids)):
        token_surprisals.append(-log_probs[t - 1, ids[t]].item())

    words = sentence_text.split()

    # Word-boundary alignment via the BPE leading-space marker ('Ġ'), the
    # same method extract_nmt_surprisal_soft.py uses for MarianTokenizer's
    # '▁' marker. Character-offset matching is NOT used here: GPT2TokenizerFast's
    # offset_mapping includes the leading space in a token's span (e.g. the
    # token for "is" in "There is" has offset covering " is", not "is"), which
    # silently misattributes every token to the PRECEDING word when matched
    # against word-start positions computed without the leading space.
    toks = tokenizer.convert_ids_to_tokens(ids)
    word_surprisals = {i: [] for i in range(len(words))}
    wi = -1
    for tok_idx, tk in enumerate(toks):
        if tok_idx == 0 or tk.startswith("Ġ"):
            wi += 1
        if wi >= len(words):
            break  # guard against tokenizer producing more boundaries than words
        word_surprisals[wi].append(token_surprisals[tok_idx])

    rows = []
    for wi, word in enumerate(words):
        sl = word_surprisals[wi]
        rows.append({
            "word_index":     wi,
            "word":           word,
            "surprisal_sum":  round(sum(sl), 6)           if sl else None,
            "surprisal_mean": round(sum(sl) / len(sl), 6) if sl else None,
            "n_tokens":       len(sl),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract GPT-2 monolingual surprisal.")
    parser.add_argument("--sentences", required=True,
                        help="Path to Sentences.csv from the EMMT corpus")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Hugging Face model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Pinned Hugging Face model revision")
    args = parser.parse_args()

    print(f"Loading {args.model} at revision {args.revision} …")
    tokenizer = GPT2TokenizerFast.from_pretrained(
        args.model, revision=args.revision
    )
    model = GPT2LMHeadModel.from_pretrained(
        args.model, revision=args.revision
    )
    model.eval()
    print("  Model loaded.\n")

    sentences = load_sentences(args.sentences)
    print(f"Loaded {len(sentences)} sentences.\n")

    all_rows = []
    for sid in sorted(sentences.keys()):
        text = sentences[sid]
        print(f"  {sid}: {text[:65]}")
        try:
            rows = compute_surprisal(text, tokenizer, model)
            for r in rows:
                r["sentence_id"] = sid
            all_rows.extend(rows)
        except Exception as e:
            print(f"    ERROR: {e}")

    fields = ["sentence_id", "word_index", "word",
              "surprisal_sum", "surprisal_mean", "n_tokens"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} word rows → {args.output}")


if __name__ == "__main__":
    main()
