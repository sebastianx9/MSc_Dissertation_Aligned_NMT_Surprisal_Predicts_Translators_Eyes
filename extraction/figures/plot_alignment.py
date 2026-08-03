#!/usr/bin/env python3
"""
Generate cross-attention alignment heatmap for paper figure.
Source: Helsinki-NLP/opus-mt-en-cs  (EN → CS)
Figure: word-level attention weights (CS tokens × EN words),
        argmax alignment highlighted with red border.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import MarianMTModel, MarianTokenizer

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-cs"
DEFAULT_REVISION = "2820c6a540ddc2b7c4ea4c95c39b3150bd3ac27e"
DEFAULT_SENTENCE = "A man is blowing into a plastic ball."

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--revision", default=DEFAULT_REVISION)
parser.add_argument("--local-files-only", action="store_true")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {args.model} …")
tokenizer = MarianTokenizer.from_pretrained(
    args.model, revision=args.revision, local_files_only=args.local_files_only
)
model = MarianMTModel.from_pretrained(
    args.model,
    revision=args.revision,
    output_attentions=True,
    local_files_only=args.local_files_only,
)
model.eval()

# ── Tokenise & generate ───────────────────────────────────────────────────────
enc     = tokenizer([args.sentence], return_tensors="pt", truncation=True, max_length=128)
src_ids = enc["input_ids"][0]
src_tokens = tokenizer.convert_ids_to_tokens(src_ids.tolist())

with torch.no_grad():
    gen_ids = model.generate(
        enc["input_ids"],
        attention_mask=enc["attention_mask"],
        num_beams=4, max_length=200,
    )[0]

# Teacher-forced forward pass
decoder_input_ids = gen_ids[:-1].unsqueeze(0)
target_ids        = gen_ids[1:].unsqueeze(0)

with torch.no_grad():
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        output_attentions=True,
    )

# Last decoder layer cross-attention, mean over heads → (T-1, S)
attn_mean = out.cross_attentions[-1][0].mean(dim=0).cpu().numpy()

cs_tokens_str = tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
print(f"Czech tokens: {cs_tokens_str}")

# ── Word-level attention matrix ───────────────────────────────────────────────
def subwords_to_word_map(tokens):
    word_map, wi = [], -1
    for tok in tokens:
        if tok in ("<pad>", "</s>", "<unk>"):
            word_map.append(-1); continue
        if tok.startswith("▁") or wi == -1:
            wi += 1
        word_map.append(wi)
    return word_map

en_words    = args.sentence.split()
en_word_map = subwords_to_word_map(src_tokens)
n_en_words  = len(en_words)
n_cs        = len(cs_tokens_str)

word_attn = np.zeros((n_cs, n_en_words))
for sub_idx, wi in enumerate(en_word_map):
    if 0 <= wi < n_en_words and sub_idx < attn_mean.shape[1]:
        word_attn[:, wi] += attn_mean[:, sub_idx]

# Filter special tokens
valid = [i for i, t in enumerate(cs_tokens_str) if t not in ("</s>", "<pad>", "<unk>")]
word_attn_f = word_attn[valid]
cs_labels   = [cs_tokens_str[i].replace("▁", "") for i in valid]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(6, n_en_words * 0.95), max(3, len(cs_labels) * 0.52 + 1.2)))
im = ax.imshow(word_attn_f, aspect="auto", cmap="Blues", vmin=0, vmax=word_attn_f.max())

ax.set_xticks(range(n_en_words))
ax.set_xticklabels(en_words, rotation=35, ha="right", fontsize=10)
ax.set_yticks(range(len(cs_labels)))
ax.set_yticklabels(cs_labels, fontsize=9)
ax.set_xlabel("English source words", fontsize=10, labelpad=6)
ax.set_ylabel("Czech target tokens", fontsize=10)
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax.set_xticks(range(n_en_words))
ax.set_xticklabels(en_words, rotation=35, ha="left", fontsize=10)

# Normalised row weights shown as text in each cell
row_sums = word_attn_f.sum(axis=1, keepdims=True)
norm_weights = word_attn_f / np.where(row_sums > 0, row_sums, 1)
for t in range(len(cs_labels)):
    for w in range(n_en_words):
        val = norm_weights[t, w]
        if val > 0.05:  # only label cells with non-trivial weight
            ax.text(w, t, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if val > 0.45 else "black")

plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Mean cross-attention weight")
plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight", dpi=300)
print(f"\nSaved: {args.output}")
