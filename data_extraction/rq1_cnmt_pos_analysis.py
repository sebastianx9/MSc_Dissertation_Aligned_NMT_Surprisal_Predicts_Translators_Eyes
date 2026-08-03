"""
Systematic word-class distribution of c_nmt (Discussion: what does c_nmt measure).

Tags the 200 EMMT source sentences with spaCy POS, joins per-word c_nmt values
(same translate-stage sample as the artefact screening, n=5,152 obs / 1,793
positions, before the stoplight exclusion), and reports the c_nmt distribution
by broad word class (content vs. function words) and by detailed POS tag.

This is computational annotation of the existing public source-sentence text;
no new participant data.
"""
import csv
import spacy
import statistics as st

SENTENCES_CSV = "/Users/sebastianx/eyetracked-multi-modal-translation/probes/Sentences.csv"
NMT_CSV       = "/Users/sebastianx/Dissertation_Data/nmt_surprisal_soft_word.csv"
OUT_CSV       = "/Users/sebastianx/Dissertation RQ1/rq1_cnmt_pos.csv"

FUNCTION_POS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}
CONTENT_POS  = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

nlp = spacy.load("en_core_web_sm")

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

sentences = load_sentences(SENTENCES_CSV)
print(f"Loaded {len(sentences)} sentences.")

# word_index in nmt csv is assigned by text.split() (whitespace tokenisation,
# matching extract_nmt_surprisal_soft.py). Reproduce that exact tokenisation
# so POS tags align to the same word_index, and run spaCy over the
# whitespace tokens (not spaCy's own tokenizer) to avoid index drift.
rows = []
for sid in sorted(sentences.keys()):
    text = sentences[sid]
    words = text.split()
    doc = spacy.tokens.Doc(nlp.vocab, words=words)
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    for wi, tok in enumerate(doc):
        rows.append({"sentence_id": sid, "word_index": wi, "word": words[wi],
                      "pos": tok.pos_, "tag": tok.tag_})

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["sentence_id", "word_index", "word", "pos", "tag"])
    w.writeheader()
    w.writerows(rows)
print(f"Tagged {len(rows)} tokens -> {OUT_CSV}")

# ---- join to c_nmt and summarise ----
nmt = {}
with open(NMT_CSV, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        nmt[(r["sentence_id"], r["word_index"])] = float(r["surprisal_soft"])

vals = [nmt[(r["sentence_id"], str(r["word_index"]))]
        for r in rows if (r["sentence_id"], str(r["word_index"])) in nmt]
mean_all, sd_all = st.mean(vals), st.pstdev(vals)

def zscore(x):
    return (x - mean_all) / sd_all

by_pos = {}
by_class = {"content": [], "function": [], "other": []}
for r in rows:
    key = (r["sentence_id"], str(r["word_index"]))
    if key not in nmt:
        continue
    z = zscore(nmt[key])
    by_pos.setdefault(r["pos"], []).append(z)
    if r["pos"] in CONTENT_POS:
        by_class["content"].append(z)
    elif r["pos"] in FUNCTION_POS:
        by_class["function"].append(z)
    else:
        by_class["other"].append(z)

print(f"\nJoined n = {len(vals)} (mean={mean_all:.4f}, sd={sd_all:.4f})\n")
print("=== By broad class ===")
for k, v in by_class.items():
    if v:
        print(f"  {k:<10} n={len(v):>5}  mean_c_nmt={st.mean(v):+.3f}  median={st.median(v):+.3f}")

print("\n=== By POS tag (n >= 20) ===")
for pos, v in sorted(by_pos.items(), key=lambda kv: -st.mean(kv[1])):
    if len(v) >= 20:
        print(f"  {pos:<8} n={len(v):>5}  mean_c_nmt={st.mean(v):+.3f}  median={st.median(v):+.3f}")

# Mann-Whitney-style rank check (content vs function), reported via simple
# mean difference + bootstrap CI (content is the theoretically predicted
# LOWER group: determinate translations; function words predicted HIGHER:
# multiple admissible renderings).
import random
random.seed(42)
c, f = by_class["content"], by_class["function"]
obs_diff = st.mean(f) - st.mean(c)
boots = []
for _ in range(2000):
    cs = [random.choice(c) for _ in c]
    fs = [random.choice(f) for _ in f]
    boots.append(st.mean(fs) - st.mean(cs))
boots.sort()
lo, hi = boots[int(0.025*len(boots))], boots[int(0.975*len(boots))]
print(f"\nfunction - content mean c_nmt diff = {obs_diff:+.3f} SD, 95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]")
