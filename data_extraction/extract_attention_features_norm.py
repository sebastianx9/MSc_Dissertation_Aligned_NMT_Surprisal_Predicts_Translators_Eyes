"""
Extract Lim et al. (2024) source-side attention features WITH normalization.

Normalization follows Lim et al. Sec. 6: each raw feature is divided by its
"dummy" value — the value the feature would take under uniform attention
(a_kl = 1/N for all k,l, where N = total source sequence length).

Dummy values per feature (token level):
  H_e  (entropy renorm over ctx):   log(N_ctx)      — uniform entropy over N_ctx positions
  f_e  (flow u → context):          N_ctx / N
  f_eos (flow u → eos):             1 / N
  f_recv (flow ctx → u):            N_ctx / N
  f_cross (cross-attn sum / heads): N_tgt / N        — N_tgt = target seq len

Since normalization factors are sentence-level constants, they are applied
after word-level averaging. Raw features are identical to extract_attention_features_full.py.

Output: attention_features_6_norm.csv
"""

import csv
import math
import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

SENTENCES_CSV = "/Users/sebastianx/eyetracked-multi-modal-translation/probes/Sentences.csv"
OUTPUT_CSV    = "/Users/sebastianx/Dissertation_Data/attention_features_6_norm.csv"
MODEL_NAME    = "Helsinki-NLP/opus-mt-en-cs"

print(f"Loading {MODEL_NAME} …")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()
print("  Model loaded.\n")


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


def entropy_renorm(attn_row, ctx_indices):
    ctx_weights = [attn_row[j] for j in ctx_indices]
    total = sum(ctx_weights)
    if total < 1e-9:
        return 0.0
    h = 0.0
    for w in ctx_weights:
        p = w / total
        if p > 1e-9:
            h -= p * math.log(p)
    return h


def extract_features(sentence_text):
    enc     = tokenizer([sentence_text], return_tensors="pt",
                        truncation=True, max_length=128)
    src_ids = enc["input_ids"][0]
    src_toks = tokenizer.convert_ids_to_tokens(src_ids.tolist())
    word_map = subwords_to_word_map(src_toks)
    n_words  = max(wi for wi in word_map if wi >= 0) + 1
    seq_len  = len(src_toks)                        # N (source)

    eos_idx      = next((i for i, t in enumerate(src_toks) if t == "</s>"), None)
    ctx_positions = {j for j, wi in enumerate(word_map) if wi >= 0}
    N_ctx = len(ctx_positions) - 1                  # context size (excl. self)

    # ── Encoder self-attention ────────────────────────────────────────────────
    with torch.no_grad():
        enc_out = model.model.encoder(
            input_ids         = enc["input_ids"],
            attention_mask    = enc["attention_mask"],
            output_attentions = True,
        )

    n_layers = len(enc_out.attentions)
    n_heads  = enc_out.attentions[0].shape[1]
    count    = n_layers * n_heads

    token_entropy  = [0.0] * seq_len
    token_ctx_att  = [0.0] * seq_len
    token_eos_att  = [0.0] * seq_len
    token_recv_att = [0.0] * seq_len

    for layer_attn in enc_out.attentions:
        attn = layer_attn[0]
        for h in range(n_heads):
            for i in range(seq_len):
                row = attn[h, i].tolist()
                ctx_indices = [j for j in ctx_positions if j != i]
                token_entropy[i] += entropy_renorm(row, ctx_indices)
                token_ctx_att[i] += sum(row[j] for j in ctx_indices)
                if eos_idx is not None:
                    token_eos_att[i] += row[eos_idx]

            for j in range(seq_len):
                if j not in ctx_positions:
                    continue
                col = attn[h, :, j].tolist()
                for i in ctx_positions:
                    if i != j:
                        token_recv_att[j] += col[i]

    token_entropy  = [v / count for v in token_entropy]
    token_ctx_att  = [v / count for v in token_ctx_att]
    token_eos_att  = [v / count for v in token_eos_att]
    token_recv_att = [v / count for v in token_recv_att]

    # ── Cross-attention ───────────────────────────────────────────────────────
    with torch.no_grad():
        gen_ids = model.generate(
            enc["input_ids"],
            attention_mask = enc["attention_mask"],
            num_beams = 4, max_length = 200,
        )[0]

    decoder_input_ids = gen_ids[:-1].unsqueeze(0)
    N_tgt = decoder_input_ids.shape[1]              # target seq len

    with torch.no_grad():
        out = model(
            input_ids         = enc["input_ids"],
            attention_mask    = enc["attention_mask"],
            decoder_input_ids = decoder_input_ids,
            output_attentions = True,
        )

    n_cross_layers = len(out.cross_attentions)
    n_cross_heads  = out.cross_attentions[0].shape[1]
    cross_count    = n_cross_layers * n_cross_heads

    token_cross = np.zeros(seq_len)
    for layer_ca in out.cross_attentions:
        ca = layer_ca[0].cpu().numpy()
        for h in range(n_cross_heads):
            token_cross += ca[h].sum(axis=0)
    token_cross = token_cross / cross_count

    # ── Dummy values (uniform attention) ─────────────────────────────────────
    # H_e  dummy: log(N_ctx)        — entropy of uniform dist over N_ctx positions
    # f_e  dummy: N_ctx / seq_len
    # f_eos dummy: 1 / seq_len
    # f_recv dummy: N_ctx / seq_len
    # f_cross dummy: N_tgt / seq_len
    log_N_ctx  = math.log(max(N_ctx, 2))            # guard against N_ctx < 2
    dummy_fe   = N_ctx / seq_len
    dummy_feos = 1.0 / seq_len
    dummy_recv = N_ctx / seq_len
    dummy_fc   = N_tgt / seq_len

    # ── Aggregate to word level, then normalise ───────────────────────────────
    accum = {wi: {"entropy": 0.0, "ctx": 0.0, "eos": 0.0,
                  "recv": 0.0, "cross": 0.0, "n": 0}
             for wi in range(n_words)}

    for tok_i, wi in enumerate(word_map):
        if wi < 0:
            continue
        accum[wi]["entropy"] += token_entropy[tok_i]
        accum[wi]["ctx"]     += token_ctx_att[tok_i]
        accum[wi]["eos"]     += token_eos_att[tok_i]
        accum[wi]["recv"]    += token_recv_att[tok_i]
        accum[wi]["cross"]   += token_cross[tok_i]
        accum[wi]["n"]       += 1

    words = sentence_text.split()
    rows  = []
    for wi, word in enumerate(words):
        n = accum[wi]["n"]
        if n == 0:
            rows.append({"word_index": wi, "word": word,
                         "attn_entropy": None, "attn_context": None,
                         "attn_eos": None, "attn_recv": None, "attn_cross": None})
        else:
            # raw word-level means
            H_e_raw    = accum[wi]["entropy"] / n
            f_e_raw    = accum[wi]["ctx"]     / n
            f_eos_raw  = accum[wi]["eos"]     / n
            f_recv_raw = accum[wi]["recv"]    / n
            f_cross_raw= accum[wi]["cross"]   / n

            # normalise: divide by dummy value
            rows.append({
                "word_index":   wi,
                "word":         word,
                "attn_entropy": round(H_e_raw    / log_N_ctx,  6),
                "attn_context": round(f_e_raw    / dummy_fe,   6),
                "attn_eos":     round(f_eos_raw  / dummy_feos, 6),
                "attn_recv":    round(f_recv_raw / dummy_recv, 6),
                "attn_cross":   round(f_cross_raw/ dummy_fc,   6),
            })
    return rows


def main():
    sentences = load_sentences(SENTENCES_CSV)
    print(f"Loaded {len(sentences)} sentences.\n")

    all_rows = []
    for sid in sorted(sentences.keys()):
        text = sentences[sid]
        print(f"  {sid}: {text[:65]}")
        try:
            rows = extract_features(text)
            for r in rows:
                r["sentence_id"] = sid
            all_rows.extend(rows)
        except Exception as e:
            print(f"    ERROR: {e}")

    fields = ["sentence_id", "word_index", "word",
              "attn_entropy", "attn_context", "attn_eos", "attn_recv", "attn_cross"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} word rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
