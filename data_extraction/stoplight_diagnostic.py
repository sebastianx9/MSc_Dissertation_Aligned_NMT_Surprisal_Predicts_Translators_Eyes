"""Diagnostic: S003 with 'stoplight' vs 'stop light' through the same
soft-aligned NMT surprisal pipeline (extract_nmt_surprisal_soft.py logic)."""
import torch
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-cs"
DEVICE = torch.device("cpu")

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval().to(DEVICE)

def subwords_to_word_map(tokens):
    word_map, wi = [], -1
    for tok in tokens:
        if tok in ("<pad>", "</s>", "<unk>"):
            word_map.append(-1); continue
        if tok.startswith("▁") or wi == -1:
            wi += 1
        word_map.append(wi)
    return word_map

def process_sentence(text):
    enc = tokenizer([text], return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    src_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    with torch.no_grad():
        gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                 num_beams=4, max_length=200)[0]
    translation = tokenizer.decode(gen_ids, skip_special_tokens=True)
    decoder_input_ids = gen_ids[:-1].unsqueeze(0)
    target_ids = gen_ids[1:].unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    decoder_input_ids=decoder_input_ids, output_attentions=True)
    log_probs = torch.nn.functional.log_softmax(out.logits[0], dim=-1)
    surprisals = [-log_probs[t, tok].item() for t, tok in enumerate(target_ids[0].tolist())]
    attn_mean = out.cross_attentions[-1][0].mean(dim=0).cpu().numpy()
    cs_tokens = tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
    en_words = text.split()
    en_word_map = subwords_to_word_map(src_tokens)
    n_en, n_cs = len(en_words), len(cs_tokens)
    word_attn = np.zeros((n_cs, n_en))
    for sub_idx, wi in enumerate(en_word_map):
        if 0 <= wi < n_en and sub_idx < attn_mean.shape[1]:
            word_attn[:, wi] += attn_mean[:, sub_idx]
    soft_surp = np.zeros(n_en)
    for t, (tok, surp) in enumerate(zip(cs_tokens, surprisals)):
        if tok in ("</s>", "<pad>", "<unk>"): continue
        row = word_attn[t]; s = row.sum()
        if s > 0:
            soft_surp += surp * (row / s)
    return translation, list(zip(en_words, soft_surp.round(3)))

for label, text in [
    ("ORIGINAL", "There is a stoplight with images drawn on each of the lights."),
    ("REWRITTEN", "There is a stop light with images drawn on each of the lights."),
]:
    tr, words = process_sentence(text)
    print(f"\n=== {label} ===")
    print(f"  CS: {tr}")
    for w, s in words:
        print(f"    {w:<12} {s:>8.3f}")
