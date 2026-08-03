# Aligned Translation Surprisal and Source-Text Eye Movements

Code for an MSc dissertation on whether neural machine translation (NMT)
signals predict English source-text fixation durations during English--Czech
sight translation. Eye-tracking data come from the EMMT corpus (Bhattacharya
et al., 2022).

Aligned NMT surprisal, $c_\mathrm{nmt}$, is the surprisal of the NMT model's
generated Czech tokens distributed back to English source words through soft
cross-attention alignment. Monolingual surprisal, $c_\mathrm{mono}$, is summed
GPT-2 subword surprisal for each English source word.

## Research questions

- **RQ1:** Does $c_\mathrm{nmt}$ predict source-text fixation duration during
  sight translation, and how does its held-out gain compare with
  $c_\mathrm{mono}$ and five NMT attention features?
- **RQ2 (exploratory):** Is the $c_\mathrm{nmt}$ slope larger during
  translation than during prior oral reading, after lexical and positional
  controls and $c_\mathrm{mono}$ are included jointly?
- **RQ3 (exploratory):** Which part of source-text processing carries the
  association? Re-reading time is conditional on a word having been
  revisited; the probability of regression-in is not modelled.

Because reading always precedes translation, RQ2 identifies a
translation-related stage difference within this fixed-order paradigm. It
does not isolate a translation-specific causal effect from task, repetition,
and order.

## Headline results

All predictive comparisons use the same sentence-grouped 10-fold allocation,
sentence-clustered standard errors, and sentence-level sign-flip tests.

| Comparison | Delta ELPD | Clustered SE | p |
|---|---:|---:|---:|
| $c_\mathrm{nmt}$ vs controls | +14.48 | 5.65 | .038 Holm |
| $c_\mathrm{mono}$ vs controls | +4.80 | 2.77 | .042 raw; .25 Holm |
| $M_\mathrm{nmt}-M_\mathrm{mono}$, paired directly | +9.67 | 5.59 | .047 one-sided; .095 two-sided |
| $c_\mathrm{nmt}$ beyond alignment mass | +9.15 | 4.99 | .031 |
| $c_\mathrm{nmt}$ with *stoplight* retained | +10.99 | 5.61 | .027 |

The direct NMT--monolingual contrast is statistically borderline, rather than
decisive. In the RQ2 joint model, the condition-by-$c_\mathrm{nmt}$ coefficient
was 0.053 (95% credible interval [0.013, 0.092]); the implied slopes were 0.013
[-0.017, 0.044] during reading and 0.066 [0.032, 0.101] during translation.
On the translation stage, $c_\mathrm{nmt}$ also improved prediction beyond a
baseline already containing $c_\mathrm{mono}$ (Delta ELPD = +11.36,
clustered SE = 5.09, p = .015).

For RQ3, gains were -1.22 for first-fixation duration, +0.59 for gaze duration,
and +8.46 for conditional re-reading time (clustered SE = 4.54, nominal
one-sided p = .032). This is tentative evidence about re-reading duration
among revisited words, not evidence that higher $c_\mathrm{nmt}$ makes a word
more likely to be revisited. Non-significant first-pass comparisons are not
equivalence tests.

## Data

The EMMT recordings are available from the
[UFAL EMMT repository](https://github.com/ufal/eyetracked-multi-modal-translation)
and are not redistributed here. SUBTLEX-US supplies the frequency norms.
Expected analysis inputs are:

- `fixation_durations_word.csv`
- `eye_measures_word.csv`
- `nmt_surprisal_soft_word.csv`
- `nmt_alignment_mass_word.csv`
- `monolingual_surprisal_word.csv`
- `attention_features_6_norm.csv`
- `subtlex_us.csv`

Set `DATA_DIR` near the top of the R scripts, or pass `--data-dir=PATH` to the
newer command-line scripts. Model caches and derived CSVs are deliberately not
tracked.

## Reproducing feature extraction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data-extraction/extract_nmt_surprisal_soft.py \
  --sentences /path/to/EMMT/probes/Sentences.csv \
  --output /path/to/data/nmt_surprisal_soft_word.csv

python data-extraction/extract_nmt_alignment_mass.py \
  --sentences /path/to/EMMT/probes/Sentences.csv \
  --output /path/to/data/nmt_alignment_mass_word.csv

python data-extraction/extract_monolingual_surprisal.py \
  --sentences /path/to/EMMT/probes/Sentences.csv \
  --output /path/to/data/monolingual_surprisal_word.csv
```

The extractors default to the revisions used for the dissertation:

- `Helsinki-NLP/opus-mt-en-cs`:
  `2820c6a540ddc2b7c4ea4c95c39b3150bd3ac27e`
- `gpt2`: `607a30d783dfa663caf39e06633721c8d4cfcd7e`

The monolingual extractor aligns GPT-2 tokens through the leading-space BPE
marker (`Ġ`). This replaces the earlier offset-based implementation, which
could assign a token spanning a leading space to the preceding word.

## Main analysis scripts

```text
RQ1/rq1_kfold_elpd.R                  seven predictors vs the shared baseline
RQ1/rq1_direct_nmt_vs_mono.R          paired NMT-minus-monolingual ELPD test
RQ1/rq1_mass_stoplight_robustness.R   alignment-mass and stoplight checks
RQ1/rq_locus_kfold.R                  current/preceding/following c_mono check

RQ2/rq2_joint_maximal.R               joint stage-interaction model
RQ2/rq2_kfold_elpd.R                  nested predictive comparisons
RQ2/rq2_stoplight_importance.R        sequential stoplight sensitivity check

RQ3/rq3_kfold_elpd.R                  FFD, GD, and conditional-RRT comparisons
RQ3/rq3_gd_rrt.R                      RQ3 figure from saved model results
```

The complete directories also contain diagnostic, plotting, and alternative
specification scripts. The previous README's “known gap” no longer applies:
the exact sentence-grouped RQ1 and RQ3 scripts are now included.

## Computational environment

The recorded environment was:

- Python 3.13.3; PyTorch 2.11.0; Transformers 5.6.2; NumPy 2.4.3;
  pandas 3.0.1
- R 4.5.1; brms 2.23.0; loo 2.9.0; lme4 2.0.1; lmerTest 3.2.1;
  dplyr 1.2.1; posterior 1.7.0

Random seed 42 fixes the primary grouped folds and sign-flip tests. The main
predictor families use 10,000 sign flips; the direct NMT--monolingual contrast
and the alignment-mass and *stoplight* checks use 1,000.

## References

- Bhattacharya, S., Kloudova, V., Zouhar, V., & Bojar, O. (2022). EMMT: A
  simultaneous eye-tracking, 4-electrode EEG and audio corpus for multi-modal
  reading and translation scenarios. *arXiv:2204.02905*.
- Lim, Z. W., Vylomova, E., Kemp, C., & Cohn, T. (2024). Predicting human
  translation difficulty with neural machine translation. *TACL*, 12,
  1479--1496.
- Wilcox, E. G. et al. (2023). Testing the predictions of surprisal theory in
  11 languages. *TACL*.
