#!/usr/bin/env Rscript

# Fail-fast CSF check: packages, corrected input data, and a tiny Stan fit.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  hit <- grep(paste0("^", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", name, "="), "", hit[[1]])
}
data_dir <- normalizePath(
  get_arg("--data-dir", Sys.getenv("DISSERTATION_DATA_DIR", ".")),
  mustWork = TRUE
)
skip_stan <- tolower(get_arg("--skip-stan", "false")) == "true"

required_packages <- c(
  "brms", "dplyr", "loo", "posterior",
  "bayesplot", "ggplot2", "gridExtra"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Missing R packages: ", paste(missing_packages, collapse = ", "),
    "\nRun hpc/csf3_install_packages.sbatch, then resubmit this check."
  )
}
suppressPackageStartupMessages(library(dplyr))

script_file <- sub(
  "^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
)
repo_dir <- Sys.getenv(
  "DISSERTATION_REPO_DIR",
  normalizePath(file.path(dirname(script_file), ".."), mustWork = TRUE)
)
source(file.path(repo_dir, "analysis", "shared", "analysis_design.R"))

registry_path <- file.path(repo_dir, "config", "analyses.tsv")
registry <- read.delim(
  registry_path, sep = "\t", quote = "", comment.char = "",
  stringsAsFactors = FALSE, check.names = FALSE
)
stopifnot(
  identical(names(registry), c("analysis", "script", "fixed_args")),
  !anyDuplicated(registry$analysis),
  all(nzchar(registry$analysis)),
  all(nzchar(registry$script))
)
analysis_scripts <- file.path(repo_dir, unique(registry$script))
stopifnot(all(file.exists(analysis_scripts)))
invisible(lapply(analysis_scripts, parse))
cat(sprintf("Parsed %d authoritative analysis scripts successfully.\n",
            length(analysis_scripts)))

required_files <- c(
  "fixation_durations_word.csv", "eye_measures_word.csv",
  "fixation_durations_word_line_diagnostics.csv",
  "eye_measures_word_line_diagnostics.csv",
  "nmt_surprisal_soft_word.csv", "nmt_alignment_mass_word.csv",
  "monolingual_surprisal_word.csv", "attention_features_6_norm.csv",
  "subtlex_us.csv"
)
missing_files <- required_files[
  !file.exists(file.path(data_dir, required_files))
]
if (length(missing_files)) {
  stop("Missing analysis inputs: ", paste(missing_files, collapse = ", "))
}

fix <- read.csv(
  file.path(data_dir, "fixation_durations_word.csv"),
  stringsAsFactors = FALSE
)
eye <- read.csv(
  file.path(data_dir, "eye_measures_word.csv"),
  stringsAsFactors = FALSE
)
fix_diag <- read.csv(
  file.path(data_dir, "fixation_durations_word_line_diagnostics.csv"),
  stringsAsFactors = FALSE
)
eye_diag <- read.csv(
  file.path(data_dir, "eye_measures_word_line_diagnostics.csv"),
  stringsAsFactors = FALSE
)
word_key <- c(
  "participant", "order", "sentence_id", "ambiguity", "congruency",
  "stage", "word_index", "word"
)
stopifnot(
  nrow(fix) == nrow(eye),
  nrow(fix) == 19857L,
  all(c(
    "go_past_ms", "go_past_status", "first_encounter_status",
    "reread_occurrence", "line_fit_source"
  ) %in% names(eye)),
  "line_fit_source" %in% names(fix),
  identical(fix[word_key], eye[word_key]),
  identical(fix$line_fit_source, eye$line_fit_source),
  isTRUE(all.equal(
    fix$total_fixation_duration_ms, eye$tfd_ms, tolerance = 0
  )),
  sum(eye$stage == "read" & !is.na(eye$go_past_ms)) == 6081L,
  sum(eye$stage == "translate" & !is.na(eye$go_past_ms)) == 4882L,
  length(unique(fix$participant)) == 40L,
  setequal(
    setdiff(sprintf("P%02d", 1:43), unique(fix$participant)),
    c("P10", "P14", "P38")
  ),
  n_distinct(fix$sentence_id) == 200L,
  n_distinct(eye$sentence_id) == 200L,
  sum(fix$stage == "read") == 10751L,
  sum(fix$stage == "translate") == 9106L,
  setequal(
    intersect(unique(fix$sentence_id), c("S031", "S032")),
    c("S031", "S032")
  ),
  setequal(
    intersect(unique(eye$sentence_id), c("S031", "S032")),
    c("S031", "S032")
  )
)
diagnostic_key <- c(
  "participant", "order", "sentence_id", "ambiguity", "congruency",
  "stage", "file"
)
trial_key <- setdiff(diagnostic_key, "file")
accepted_trial_keys <- fix_diag %>%
  filter(status == "ok") %>%
  distinct(across(all_of(trial_key)))
word_trial_keys <- fix %>%
  distinct(across(all_of(trial_key)))
prior_diagnostics <- fix_diag %>%
  filter(status == "ok", fit_source == "read_prior")
stopifnot(
  nrow(fix_diag) == 2746L,
  identical(fix_diag[diagnostic_key], eye_diag[diagnostic_key]),
  isTRUE(all.equal(fix_diag, eye_diag, tolerance = 0)),
  sum(fix_diag$status == "ok") == 2459L,
  sum(fix_diag$status == "ok" & fix_diag$stage == "read") == 1233L,
  sum(fix_diag$status == "ok" & fix_diag$stage == "translate") == 1226L,
  sum(fix_diag$status == "ok" & fix_diag$fit_source == "read_prior") == 190L,
  nrow(anti_join(accepted_trial_keys, word_trial_keys, by = trial_key)) == 0L,
  nrow(anti_join(word_trial_keys, accepted_trial_keys, by = trial_key)) == 0L,
  all(abs(prior_diagnostics$beta - prior_diagnostics$read_beta) < 1e-4),
  all(abs(prior_diagnostics$translation_read_shift_px) <= 80),
  all(prior_diagnostics$mode_score_ratio >= 2),
  sum(grepl(
    "prior_membership_cycle_resolved", fix_diag$review_flags, fixed = TRUE
  ), na.rm = TRUE) == 1L,
  all(
    fix_diag$n_mapped_word_bouts + fix_diag$n_offtext_bouts +
      fix_diag$n_unknown_bouts == fix_diag$n_raw_bouts
  )
)
cat(sprintf(
  paste0(
    "Line-corrected full inputs: %d rows | %d participants | %d sentence IDs | ",
    "read=%d | translate=%d\n"
  ),
  nrow(fix), length(unique(fix$participant)),
  n_distinct(fix$sentence_id),
  sum(fix$stage == "read"), sum(fix$stage == "translate")
))

fix_without_contrastive <- fix %>%
  filter(!sentence_id %in% c("S031", "S032"))
eye_without_contrastive <- eye %>%
  filter(!sentence_id %in% c("S031", "S032"))
stopifnot(
  nrow(fix_without_contrastive) == 18351L,
  nrow(eye_without_contrastive) == 18351L,
  n_distinct(fix_without_contrastive$sentence_id) == 198L,
  n_distinct(eye_without_contrastive$sentence_id) == 198L
)
cat(sprintf(
  "Leave-pair-out subset: %d fixation rows | %d eye-measure rows | 198 sentences\n",
  nrow(fix_without_contrastive), nrow(eye_without_contrastive)
))

nmt <- read.csv(
  file.path(data_dir, "nmt_surprisal_soft_word.csv"),
  stringsAsFactors = FALSE
)
mono <- read.csv(
  file.path(data_dir, "monolingual_surprisal_word.csv"),
  stringsAsFactors = FALSE
)
attn <- read.csv(
  file.path(data_dir, "attention_features_6_norm.csv"),
  stringsAsFactors = FALSE
)
required_attention_columns <- c(
  "sentence_id", "word_index", "word", "attn_entropy", "attn_context",
  "attn_self", "attn_eos", "attn_recv", "attn_cross"
)
missing_attention_columns <- setdiff(required_attention_columns, names(attn))
if (length(missing_attention_columns)) {
  stop(
    "attention_features_6_norm.csv is not the six-feature Lim-style file; ",
    "missing columns: ", paste(missing_attention_columns, collapse=", ")
  )
}
attention_values <- as.matrix(attn[
  c("attn_entropy", "attn_context", "attn_self", "attn_eos",
    "attn_recv", "attn_cross")
])
stopifnot(
  nrow(attn) == 2168L,
  !anyDuplicated(attn[c("sentence_id", "word_index")]),
  all(is.finite(attention_values))
)
freq <- read.table(
  file.path(data_dir, "subtlex_us.csv"), sep = "\t", header = TRUE,
  stringsAsFactors = FALSE, quote = ""
) %>%
  transmute(word_lower = lexical_form(Word), log10_freq = Lg10WF)

sentence_lengths <- nmt %>%
  group_by(sentence_id) %>%
  summarise(sentence_length = max(word_index) + 1, .groups = "drop")
predictors <- nmt %>%
  left_join(sentence_lengths, by = "sentence_id") %>%
  mutate(
    word_lower = lexical_form(word),
    word_length = nchar(word_lower, type = "chars"),
    word_position = word_index / (sentence_length - 1)
  ) %>%
  left_join(freq, by = "word_lower") %>%
  left_join(
    mono %>% select(
      sentence_id, word_index, mono_surprisal = surprisal_sum
    ),
    by = c("sentence_id", "word_index")
  ) %>%
  left_join(
    attn %>% select(
      sentence_id, word_index, H_e = attn_entropy,
      f_e = attn_context, f_self = attn_self, f_eos = attn_eos,
      f_recv = attn_recv, f_cross = attn_cross
    ),
    by = c("sentence_id", "word_index")
  )

primary <- fix %>%
  filter(stage == "translate") %>%
  left_join(predictors, by = c("sentence_id", "word_index")) %>%
  filter(
    !is.na(surprisal_soft), !is.na(mono_surprisal), !is.na(log10_freq),
    !is.na(H_e), !is.na(f_e), !is.na(f_self), !is.na(f_eos), !is.na(f_recv),
    !is.na(f_cross)
  ) %>%
  anti_join(tibble(sentence_id = "S003", word_index = 3L),
            by = c("sentence_id", "word_index"))
pooled <- fix %>%
  filter(stage %in% c("translate", "read")) %>%
  left_join(predictors, by = c("sentence_id", "word_index")) %>%
  filter(
    !is.na(surprisal_soft), !is.na(mono_surprisal), !is.na(log10_freq)
  ) %>%
  anti_join(tibble(sentence_id = "S003", word_index = 3L),
            by = c("sentence_id", "word_index"))
rrt <- eye %>%
  filter(stage == "translate") %>%
  left_join(predictors, by = c("sentence_id", "word_index")) %>%
  filter(
    !is.na(surprisal_soft), !is.na(mono_surprisal), !is.na(log10_freq),
    !(sentence_id == "S003" & word_index == 3L),
    reread_occurrence == 1L, rrt_ms > 0
  )
go_past <- eye %>%
  filter(stage == "translate") %>%
  left_join(predictors, by = c("sentence_id", "word_index")) %>%
  filter(
    !is.na(surprisal_soft), !is.na(mono_surprisal), !is.na(log10_freq),
    !(sentence_id == "S003" & word_index == 3L),
    !is.na(go_past_ms), go_past_ms > 0
  )
stopifnot(
  nrow(primary) == 9047L,
  nrow(pooled) == 19732L,
  sum(pooled$stage == "translate") == 9047L,
  sum(pooled$stage == "read") == 10685L,
  nrow(rrt) == 4789L,
  nrow(go_past) == 4851L,
  n_distinct(primary$sentence_id) == 200L,
  n_distinct(pooled$sentence_id) == 200L
)
primary_design <- design_counts(primary$sentence_id)
stopifnot(
  unname(primary_design[["n_sentence_ids"]]) == 200L,
  unname(primary_design[["n_inference_clusters"]]) == 199L
)
primary_folds <- make_sentence_folds(primary$sentence_id, K = 10L, seed = 42L)
assert_contrastive_fold_binding(primary_folds, primary$sentence_id)
cat(sprintf(
  paste0(
    "Analysis samples: RQ1=%d/%d sentences | ",
    "RQ2=%d (%d translate + %d read) | go-past=%d | conditional RRT=%d\n"
  ),
  nrow(primary), n_distinct(primary$sentence_id), nrow(pooled),
  sum(pooled$stage == "translate"), sum(pooled$stage == "read"),
  nrow(go_past),
  nrow(rrt)
))

primary_without_contrastive <- primary %>%
  filter(!sentence_id %in% c("S031", "S032"))
pooled_without_contrastive <- pooled %>%
  filter(!sentence_id %in% c("S031", "S032"))
rrt_without_contrastive <- rrt %>%
  filter(!sentence_id %in% c("S031", "S032"))
go_past_without_contrastive <- go_past %>%
  filter(!sentence_id %in% c("S031", "S032"))
stopifnot(
  nrow(primary_without_contrastive) == 8328L,
  nrow(pooled_without_contrastive) == 18226L,
  sum(pooled_without_contrastive$stage == "translate") == 8328L,
  sum(pooled_without_contrastive$stage == "read") == 9898L,
  nrow(rrt_without_contrastive) == 4406L,
  nrow(go_past_without_contrastive) == 4397L,
  n_distinct(primary_without_contrastive$sentence_id) == 198L,
  n_distinct(pooled_without_contrastive$sentence_id) == 198L
)
excluded_design <- design_counts(primary_without_contrastive$sentence_id)
stopifnot(
  unname(excluded_design[["n_sentence_ids"]]) == 198L,
  unname(excluded_design[["n_inference_clusters"]]) == 198L
)
cat(sprintf(
  paste0(
    "Leave-pair-out analysis samples: RQ1=%d/198 sentences | ",
    "RQ2=%d (%d translate + %d read) | go-past=%d | conditional RRT=%d\n"
  ),
  nrow(primary_without_contrastive), nrow(pooled_without_contrastive),
  sum(pooled_without_contrastive$stage == "translate"),
  sum(pooled_without_contrastive$stage == "read"),
  nrow(go_past_without_contrastive),
  nrow(rrt_without_contrastive)
))

cat("\nPackage versions:\n")
for (package in required_packages) {
  cat(sprintf("  %-10s %s\n", package, as.character(packageVersion(package))))
}

if (!skip_stan) {
  # This deliberately small model validates the full brms -> Stan toolchain.
  set.seed(42)
  smoke_data <- data.frame(
    y = rnorm(40), x = rnorm(40),
    group = factor(rep(seq_len(10), each = 4))
  )
  smoke_model <- brms::brm(
    y ~ x + (1 | group), data = smoke_data,
    chains = 2, cores = 2, iter = 300, warmup = 150,
    seed = 42, refresh = 0, silent = 2
  )
  stopifnot(all(is.finite(brms::fixef(smoke_model))))
  cat("\nStan smoke fit completed successfully.\n\n")
} else {
  cat("\nStan smoke fit skipped by request.\n\n")
}
print(sessionInfo())
