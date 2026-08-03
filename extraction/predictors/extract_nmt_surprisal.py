"""
Compute soft-aligned NMT surprisal (c_nmt) for English source words.

Soft-aligned surprisal for source word w:
  c_soft(w) = sum_t  s_t * ( word_attn[t,w] / sum_j word_attn[t,j] )

Each target token's surprisal is distributed across all source words
proportionally to normalised cross-attention weights (final decoder layer,
averaged over heads).
"""

import argparse
import csv
import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer


DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-cs"
DEFAULT_REVISION = "2820c6a540ddc2b7c4ea4c95c39b3150bd3ac27e"


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


def subwords_to_word_map(tokens):
    word_map, wi = [], -1
    for tok in tokens:
        if tok in ("<pad>", "</s>", "<unk>"):
            word_map.append(-1)
            continue
        if tok.startswith("▁") or wi == -1:
            wi += 1
        word_map.append(wi)
    return word_map


def process_sentence(text, tokenizer, model, device):
    enc = tokenizer([text], return_tensors="pt", truncation=True, max_length=128).to(device)
    src_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

    with torch.no_grad():
        gen_ids = model.generate(enc["input_ids"],
                                 attention_mask=enc["attention_mask"],
                                 num_beams=4, max_length=200)[0]

    decoder_input_ids = gen_ids[:-1].unsqueeze(0)
    target_ids        = gen_ids[1:].unsqueeze(0)

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    output_attentions=True)

    log_probs  = torch.nn.functional.log_softmax(out.logits[0], dim=-1)
    surprisals = [-log_probs[t, tok].item()
                  for t, tok in enumerate(target_ids[0].tolist())]

    attn_mean = out.cross_attentions[-1][0].mean(dim=0).cpu().numpy()  # (T-1, S)

    cs_tokens   = tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
    en_words    = text.split()
    en_word_map = subwords_to_word_map(src_tokens)
    n_en = len(en_words)
    n_cs = len(cs_tokens)

    word_attn = np.zeros((n_cs, n_en))
    for sub_idx, wi in enumerate(en_word_map):
        if 0 <= wi < n_en and sub_idx < attn_mean.shape[1]:
            word_attn[:, wi] += attn_mean[:, sub_idx]

    soft_surp = np.zeros(n_en)

    for t, (tok, surp) in enumerate(zip(cs_tokens, surprisals)):
        if tok in ("</s>", "<pad>", "<unk>"):
            continue
        row     = word_attn[t]
        row_sum = row.sum()
        if row_sum > 0:
            soft_surp += surp * (row / row_sum)

    rows = []
    for wi, word in enumerate(en_words):
        rows.append({
            "word_index":     wi,
            "word":           word,
            "surprisal_soft": round(float(soft_surp[wi]), 6),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Extract soft-aligned NMT surprisal.")
    parser.add_argument("--sentences", required=True,
                        help="Path to Sentences.csv from the EMMT corpus")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Hugging Face model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Pinned Hugging Face model revision")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} …")
    tokenizer = MarianTokenizer.from_pretrained(
        args.model, revision=args.revision
    )
    model = MarianMTModel.from_pretrained(
        args.model, revision=args.revision, output_attentions=True
    )
    model.eval()
    model.to(device)

    sentences = load_sentences(args.sentences)
    print(f"Loaded {len(sentences)} sentences.\n")

    all_rows = []
    for sid in sorted(sentences.keys()):
        text = sentences[sid]
        print(f"  {sid}: {text[:60]}")
        try:
            rows = process_sentence(text, tokenizer, model, device)
            for r in rows:
                r["sentence_id"] = sid
            all_rows.extend(rows)
        except Exception as e:
            print(f"    ERROR: {e}")

    fields = ["sentence_id", "word_index", "word", "surprisal_soft"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
