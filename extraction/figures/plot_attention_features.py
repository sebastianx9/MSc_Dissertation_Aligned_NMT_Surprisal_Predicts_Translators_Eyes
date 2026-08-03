#!/usr/bin/env python3
"""
Two-panel attention-feature schematic using REAL attention matrices for
"A man is blowing into a plastic ball." (same sentence as the alignment
figure). Highlights the source word 'blowing':
  Left  (encoder self-attention, word x word incl. </s>):
        blowing's ROW    -> f_e (context cols) + f_eos (</s> col), H_e (row entropy)
        blowing's COLUMN -> f_recv (attention received from context)
  Right (cross-attention, Czech target tokens x English source words):
        blowing's COLUMN -> f_cross
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

tok = MarianTokenizer.from_pretrained(
    args.model, revision=args.revision, local_files_only=args.local_files_only
)
model = MarianMTModel.from_pretrained(
    args.model,
    revision=args.revision,
    output_attentions=True,
    local_files_only=args.local_files_only,
)
model.eval()

enc = tok([args.sentence], return_tensors="pt", truncation=True, max_length=128)
src_toks = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
words = args.sentence.split()

# word map: real words get index; </s> gets its own trailing index; <pad> = -1
def word_map_with_eos(tokens, n_words):
    wm, wi = [], -1
    for t in tokens:
        if t == "</s>":
            wm.append(n_words)          # eos as an extra column/row
        elif t in ("<pad>", "<unk>"):
            wm.append(-1)
        else:
            if t.startswith("▁") or wi == -1:
                wi += 1
            wm.append(wi)
    return wm

wm = word_map_with_eos(src_toks, len(words))
labels_enc = words + ["</s>"]
n = len(labels_enc)

# ── Encoder self-attention (mean over layers & heads), aggregate to words ──
with torch.no_grad():
    eo = model.model.encoder(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"],
                             output_attentions=True)
A = torch.stack([a[0].mean(0) for a in eo.attentions]).mean(0).numpy()  # (S,S)
S = A.shape[0]
self_mat = np.zeros((n, n)); qc = np.zeros(n)
for qi in range(S):
    wq = wm[qi]
    if wq < 0: continue
    qc[wq] += 1
    for ki in range(S):
        wk = wm[ki]
        if wk < 0: continue
        self_mat[wq, wk] += A[qi, ki]
self_mat /= np.maximum(qc[:, None], 1)

# ── Cross-attention (Czech tokens x English words) ──
with torch.no_grad():
    gen = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                         num_beams=4, max_length=200)[0]
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                decoder_input_ids=gen[:-1].unsqueeze(0), output_attentions=True)
ca = out.cross_attentions[-1][0].mean(0).numpy()          # (T, S)
cs_toks = tok.convert_ids_to_tokens(gen[1:].tolist())
keep = [i for i, t in enumerate(cs_toks) if t not in ("</s>", "<pad>", "<unk>")]
cs_labels = [cs_toks[i].replace("▁", "") for i in keep]
cross_word = np.zeros((len(keep), len(words)))
for r, ti in enumerate(keep):
    for ki in range(S):
        wk = wm[ki]
        if 0 <= wk < len(words):
            cross_word[r, wk] += ca[ti, ki]

BW = words.index("blowing")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# Three distinct highlight colours (Okabe-Ito), each contrasting its heatmap.
ROW_C   = "#D55E00"   # blowing's row (outgoing): vermillion
COL_C   = "#CC79A7"   # blowing's column (incoming from context): reddish purple
CROSS_C = "#0072B2"   # blowing's column in cross-attention: blue
SELF_C  = "#009E73"   # diagonal cell: bluish green

# Left: encoder self-attention (Blues heatmap)
imL = axL.imshow(self_mat, cmap="Blues", vmin=0, vmax=self_mat.max())
axL.set_xticks(range(n)); axL.set_xticklabels(labels_enc, rotation=45, ha="left", fontsize=8)
axL.set_yticks(range(n)); axL.set_yticklabels(labels_enc, fontsize=8)
axL.xaxis.set_ticks_position("top"); axL.xaxis.set_label_position("top")
axL.set_xlabel("key (attended-to word)", fontsize=9)
axL.set_ylabel("query (attending word)", fontsize=9)
# highlight blowing's row (outgoing) and column (incoming)
axL.add_patch(Rectangle((-0.5, BW-0.5), n, 1, fill=False, edgecolor=ROW_C, lw=2.2))
axL.add_patch(Rectangle((BW-0.5, -0.5), 1, n, fill=False, edgecolor=COL_C, lw=2.2))
axL.add_patch(Rectangle((BW-0.5, BW-0.5), 1, 1, fill=False,
                        edgecolor=SELF_C, lw=2.5))
axL.annotate("blowing's row:\n$f_e$ = context cells; $f_\\mathrm{eos}$ = </s> cell\n$H_e$ = entropy of this row",
             xy=(n-0.5, BW), xytext=(n+0.6, BW-2.4), fontsize=8, color=ROW_C,
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=ROW_C))
axL.annotate("$f_\\mathrm{self}$ = diagonal cell",
             xy=(BW, BW), xytext=(n+0.6, BW+1.1), fontsize=8,
             color=SELF_C, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=SELF_C))
axL.annotate("blowing's column:\n$f_\\mathrm{recv}$ (received\nfrom context)",
             xy=(BW, n-0.5), xytext=(BW+1.2, n+1.3), fontsize=8, color=COL_C,
             ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color=COL_C))
axL.set_title("Encoder self-attention", fontsize=12, fontweight="bold", pad=30)

# Right: cross-attention (Greens heatmap)
imR = axR.imshow(cross_word, cmap="Greens", vmin=0, vmax=cross_word.max())
axR.set_xticks(range(len(words))); axR.set_xticklabels(words, rotation=45, ha="left", fontsize=8)
axR.set_yticks(range(len(cs_labels))); axR.set_yticklabels(cs_labels, fontsize=8)
axR.xaxis.set_ticks_position("top"); axR.xaxis.set_label_position("top")
axR.set_xlabel("English source word", fontsize=9)
axR.set_ylabel("Czech target token", fontsize=9)
axR.add_patch(Rectangle((BW-0.5, -0.5), 1, len(cs_labels), fill=False, edgecolor=CROSS_C, lw=2.2))
axR.annotate("blowing's column:\n$f_\\mathrm{cross}$ (attention target\ntokens send to blowing)",
             xy=(BW, len(cs_labels)-0.5), xytext=(BW+1.0, len(cs_labels)+1.0),
             fontsize=8, color=CROSS_C, ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color=CROSS_C))
axR.set_title("Cross-attention", fontsize=12, fontweight="bold", pad=30)

fig.tight_layout()
fig.savefig(args.output, bbox_inches="tight")
print(f"Saved {args.output}")
