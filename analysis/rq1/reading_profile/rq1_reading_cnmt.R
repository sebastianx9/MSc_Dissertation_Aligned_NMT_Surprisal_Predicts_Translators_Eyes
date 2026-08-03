#!/usr/bin/env Rscript

# RQ1 supplementary oral-reading profile: c_nmt and the direct comparison
# with c_mono on the same observations and folds.
#
# The completed reading-stage c_mono validation already contains pointwise
# 10-fold elpd for the lexical/positional baseline and for baseline + c_mono.
# This script reuses those held-out scores and fits only one additional model:
#
#   baseline + c_nmt
#
# It reports three paired contrasts on exactly the same reading observations,
# folds, and sentence-template inference clusters:
#
#   1. c_nmt model - controls-only baseline (one-sided)
#   2. c_mono model - controls-only baseline (one-sided; validation cross-check)
#   3. c_nmt model - c_mono model (two-sided exploratory comparison)
#
# These within-reading contrasts provide context for RQ2; they are not a
# cross-stage interaction test.

suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
})

options(mc.cores = as.integer(Sys.getenv("MC_CORES", "4")), warn = 1)

ANALYSIS_VERSION <- "v2"
SEED <- 42L
N_SIGN_FLIPS <- 10000L

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (!length(script_arg)) stop("Cannot determine the script path from --file.")
script_file <- normalizePath(
  sub("^--file=", "", script_arg[[1]]), mustWork = TRUE
)
repo_root <- Sys.getenv("DISSERTATION_REPO_DIR", unset = "")
if (!nzchar(repo_root)) {
  repo_root <- normalizePath(
    file.path(dirname(script_file), "..", "..", ".."), mustWork = TRUE
  )
}
helper_path <- file.path(repo_root, "analysis", "shared", "analysis_design.R")
reading_validation_script <- file.path(
  repo_root, "analysis", "rq1", "reading_profile", "rq1_reading_cmono.R"
)
source(helper_path)

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
output_dir <- get_arg(
  "--output-dir", Sys.getenv("DISSERTATION_OUTPUT_DIR", data_dir)
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

exclude_contrastive <- parse_bool(
  get_arg("--exclude-contrastive", "false"),
  "--exclude-contrastive"
)
prepare_only <- parse_bool(
  get_arg("--prepare-only", "false"),
  "--prepare-only"
)

validation_name <- variant_filename(
  "rq1_reading_cmono_v3.rds", exclude_contrastive
)
validation_path <- normalizePath(
  get_arg(
    "--reading-validation-rds",
    file.path(output_dir, validation_name)
  ),
  mustWork = TRUE
)

fixation_path <- file.path(data_dir, "fixation_durations_word.csv")
nmt_path <- file.path(data_dir, "nmt_surprisal_soft_word.csv")
mono_path <- file.path(data_dir, "monolingual_surprisal_word.csv")
frequency_path <- file.path(data_dir, "subtlex_us.csv")

# These hashes reproduce the provenance declaration made by the existing
# reading-validation script.  The new model has its own, additional hash set
# below.  This prevents pointwise scores from different data versions being
# combined silently.
validation_hashes <- analysis_input_hashes(c(
  fixation = fixation_path,
  nmt_surprisal = nmt_path,
  monolingual_surprisal = mono_path,
  frequency = frequency_path,
  analysis_design = helper_path,
  analysis_script = reading_validation_script
))

validation <- readRDS(validation_path)
if (!identical(validation$metadata$analysis_version, "v3")) {
  stop("The reading c_mono profile must be the authoritative v3 output.")
}
if (!identical(validation$metadata$stage, "read")) {
  stop("The supplied validation output is not reading-stage output.")
}
if (!identical(
  isTRUE(validation$metadata$exclude_contrastive),
  isTRUE(exclude_contrastive)
)) {
  stop("The validation output and requested contrastive-pair variant differ.")
}
if (!identical(validation$metadata$input_hashes, validation_hashes)) {
  stop(
    "The reading-validation output was created from different data or code; ",
    "do not combine its pointwise scores with this run."
  )
}

fixations <- read.csv(fixation_path, stringsAsFactors = FALSE)
nmt <- read.csv(nmt_path, stringsAsFactors = FALSE)
mono <- read.csv(mono_path, stringsAsFactors = FALSE)
frequency <- read.table(
  frequency_path,
  sep = "\t", header = TRUE, stringsAsFactors = FALSE, quote = ""
) %>%
  transmute(
    word_lower = lexical_form(Word),
    log10_freq = Lg10WF
  )

sentence_lengths <- nmt %>%
  group_by(sentence_id) %>%
  summarise(sentence_length = max(word_index) + 1L, .groups = "drop")

predictors <- nmt %>%
  left_join(sentence_lengths, by = "sentence_id") %>%
  mutate(
    word_lower = lexical_form(word),
    word_length = nchar(word_lower, type = "chars"),
    word_position = word_index / (sentence_length - 1)
  ) %>%
  left_join(frequency, by = "word_lower") %>%
  left_join(
    mono %>%
      select(
        sentence_id, word_index,
        mono_surprisal = surprisal_sum
      ),
    by = c("sentence_id", "word_index")
  ) %>%
  transmute(
    sentence_id, word_index, word_length, word_position, log10_freq,
    nmt_surprisal = surprisal_soft, mono_surprisal
  )

primary_full <- fixations %>%
  filter(stage == "read") %>%
  left_join(predictors, by = c("sentence_id", "word_index")) %>%
  mutate(
    log_tfd = log(total_fixation_duration_ms),
    ambiguity = factor(ambiguity)
  ) %>%
  filter(
    is.finite(log_tfd),
    !is.na(nmt_surprisal), !is.na(mono_surprisal),
    !is.na(word_length), !is.na(word_position), !is.na(log10_freq),
    !is.na(ambiguity)
  ) %>%
  anti_join(
    tibble(sentence_id = "S003", word_index = 3L),
    by = c("sentence_id", "word_index")
  )

if (!nrow(primary_full)) stop("The primary reading-stage sample is empty.")

scaling <- validation$scaling
required_scaling <- c(
  "word_length", "word_position", "log10_freq", "mono_surprisal"
)
if (!setequal(scaling$variable, required_scaling)) {
  stop("The reading validation contains an unexpected scaling specification.")
}
z_from_validation <- function(x, variable) {
  row <- match(variable, scaling$variable)
  (x - scaling$centre[[row]]) / scaling$scale[[row]]
}

nmt_centre <- mean(primary_full$nmt_surprisal)
nmt_scale <- sd(primary_full$nmt_surprisal)
if (!is.finite(nmt_scale) || nmt_scale <= 0) {
  stop("c_nmt must have a finite, positive SD in the reading sample.")
}

primary_full <- primary_full %>%
  mutate(
    c_wlen = z_from_validation(word_length, "word_length"),
    c_wpos = z_from_validation(word_position, "word_position"),
    c_freq = z_from_validation(log10_freq, "log10_freq"),
    c_mono = z_from_validation(mono_surprisal, "mono_surprisal"),
    c_nmt = (nmt_surprisal - nmt_centre) / nmt_scale
  )

keep <- contrastive_keep(
  primary_full$sentence_id, exclude_contrastive
)
data <- primary_full[keep, , drop = FALSE]
sentence_id <- as.character(data$sentence_id)
N <- nrow(data)
counts <- design_counts(sentence_id)
J <- unname(counts[["n_sentence_ids"]])
G <- unname(counts[["n_inference_clusters"]])

if (!identical(sentence_id, validation$sentence_id)) {
  stop("Row order differs from the reading-validation pointwise output.")
}
if (!identical(N, validation$sample$n_observations)) {
  stop("Reading sample size differs from the validation output.")
}

fold_lookup <- setNames(
  validation$fold_assignment$fold,
  validation$fold_assignment$sentence_id
)
folds <- as.integer(unname(fold_lookup[sentence_id]))
if (anyNA(folds)) stop("At least one sentence has no saved fold assignment.")
assert_contrastive_fold_binding(folds, sentence_id)

pointwise_base <- as.numeric(
  validation$pointwise_elpd$base_with_position
)
pointwise_mono <- as.numeric(
  validation$pointwise_elpd$mono_with_position
)
if (length(pointwise_base) != N || length(pointwise_mono) != N) {
  stop("Saved reading pointwise elpd does not match the reconstructed sample.")
}

cat(sprintf(
  paste0(
    "Reading bridge sample: N=%d | participants=%d | sentence IDs=%d | ",
    "inference clusters=%d | contrastive pair=%s\n"
  ),
  N, n_distinct(data$participant), J, G,
  ifelse(exclude_contrastive, "excluded", "included and paired")
))
cat(sprintf("c_nmt reading scaling: centre=%.6f | SD=%.6f\n",
            nmt_centre, nmt_scale))
cat("Fold sizes:\n")
print(table(folds))

if (prepare_only) {
  cat("Preparation and baseline-pointwise alignment checks passed.\n")
  quit(save = "no", status = 0L)
}

random_effects <- "(1 | participant) + (1 | sentence_id)"
nmt_formula <- as.formula(paste(
  "log_tfd ~ c_wlen + c_wpos + c_freq + ambiguity + c_nmt +",
  random_effects
))
formula_text <- paste(
  deparse(nmt_formula, width.cutoff = 500L), collapse = " "
)

priors <- c(
  prior(normal(0, 1), class = b),
  prior(normal(6, 1), class = Intercept),
  prior(exponential(1), class = sd),
  prior(exponential(1), class = sigma)
)

model_hashes <- analysis_input_hashes(c(
  fixation = fixation_path,
  nmt_surprisal = nmt_path,
  monolingual_surprisal = mono_path,
  frequency = frequency_path,
  reading_validation = validation_path,
  analysis_design = helper_path,
  analysis_script = script_file
))

cache_dir <- file.path(data_dir, "brm_cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
cache_name <- variant_filename(
    sprintf("rq1_reading_cnmt_%s_nmt_kfold.rds", ANALYSIS_VERSION),
  exclude_contrastive
)
cache_path <- file.path(cache_dir, cache_name)

if (file.exists(cache_path)) {
  cat("Loading cached c_nmt reading k-fold result.\n")
  kfold_nmt <- readRDS(cache_path)
  assert_analysis_input_hashes(kfold_nmt, model_hashes, cache_path)
  stopifnot(
    inherits(kfold_nmt, "kfold"),
    length(kfold_nmt$pointwise[, "elpd_kfold"]) == N,
    identical(as.integer(attr(kfold_nmt, "folds")), folds),
    identical(attr(kfold_nmt, "model_formula"), formula_text)
  )
} else {
  cat("Fitting reading controls + c_nmt and running grouped 10-fold CV...\n")
  started <- proc.time()
  model <- brm(
    nmt_formula, data = data, prior = priors,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    chains = 4, iter = 4000, warmup = 2000,
    seed = SEED, silent = 2, refresh = 0
  )
  kfold_nmt <- kfold(
    model, folds = folds,
    chains = 4, iter = 4000, warmup = 2000,
    seed = SEED, silent = 2, refresh = 0
  )
  kfold_nmt <- set_analysis_input_hashes(kfold_nmt, model_hashes)
  attr(kfold_nmt, "analysis_version") <- ANALYSIS_VERSION
  attr(kfold_nmt, "model_formula") <- formula_text
  attr(kfold_nmt, "exclude_contrastive") <- exclude_contrastive
  saveRDS(kfold_nmt, cache_path)
  cat(sprintf(
    "c_nmt reading model completed in %.1f minutes.\n",
    (proc.time() - started)[["elapsed"]] / 60
  ))
}

pointwise_nmt <- as.numeric(kfold_nmt$pointwise[, "elpd_kfold"])

sign_flip_one_sided <- function(cluster_delta, n = N_SIGN_FLIPS) {
  observed <- sum(cluster_delta)
  set.seed(SEED)
  permuted <- replicate(
    n,
    sum(cluster_delta * sample(c(-1, 1), length(cluster_delta), replace = TRUE))
  )
  (1 + sum(permuted >= observed)) / (n + 1)
}

sign_flip_two_sided <- function(cluster_delta, n = N_SIGN_FLIPS) {
  observed <- abs(sum(cluster_delta))
  set.seed(SEED)
  permuted <- replicate(
    n,
    abs(sum(
      cluster_delta * sample(c(-1, 1), length(cluster_delta), replace = TRUE)
    ))
  )
  (1 + sum(permuted >= observed)) / (n + 1)
}

contrast_result <- function(name, description, pointwise_delta, tail) {
  cluster_delta <- cluster_delta_sums(pointwise_delta, sentence_id)
  estimate <- sum(pointwise_delta)
  clustered_se <- sd(cluster_delta) * sqrt(G)
  p_value <- if (tail == "one-sided") {
    sign_flip_one_sided(cluster_delta)
  } else {
    sign_flip_two_sided(cluster_delta)
  }
  tibble(
    contrast = name,
    interpretation = description,
    delta_elpd = estimate,
    sentence_clustered_se = clustered_se,
    ci_95_low = estimate - 1.96 * clustered_se,
    ci_95_high = estimate + 1.96 * clustered_se,
    p_signflip = p_value,
    tail = tail,
    per_word_delta = estimate / N,
    n_observations = N,
    n_sentence_ids = J,
    n_inference_clusters = G,
    exclude_contrastive = exclude_contrastive
  )
}

results <- bind_rows(
  contrast_result(
    "c_nmt_minus_controls",
    "Does c_nmt improve reading-stage prediction beyond lexical/positional controls?",
    pointwise_nmt - pointwise_base,
    "one-sided"
  ),
  contrast_result(
    "c_mono_minus_controls",
    "Existing c_mono reading-stage validation on the same rows and folds",
    pointwise_mono - pointwise_base,
    "one-sided"
  ),
  contrast_result(
    "c_nmt_minus_c_mono",
    "Exploratory direct comparison of the two reading-stage predictor models",
    pointwise_nmt - pointwise_mono,
    "two-sided"
  )
)

output_stem <- variant_filename(
  sprintf("rq1_reading_cnmt_%s", ANALYSIS_VERSION),
  exclude_contrastive
)
write.csv(
  results,
  file.path(output_dir, paste0(output_stem, "_results.csv")),
  row.names = FALSE
)
saveRDS(
  list(
    metadata = list(
      analysis = "RQ1 supplementary oral-reading c_nmt profile",
      analysis_version = ANALYSIS_VERSION,
      stage = "read",
      outcome = "log total fixation duration",
      seed = SEED,
      k_folds = 10L,
      n_sign_flips = N_SIGN_FLIPS,
      random_effects = random_effects,
      nmt_formula = formula_text,
      nmt_scaling = c(centre = nmt_centre, scale = nmt_scale),
      validation_source = validation_path,
      input_hashes = model_hashes
    ),
    sample = list(
      n_observations = N,
      n_participants = n_distinct(data$participant),
      n_sentence_ids = J,
      n_inference_clusters = G
    ),
    results = results,
    pointwise_elpd = list(
      base_with_position = pointwise_base,
      mono_with_position = pointwise_mono,
      nmt_with_position = pointwise_nmt
    ),
    sentence_id = sentence_id,
    folds = folds
  ),
  file.path(output_dir, paste0(output_stem, ".rds"))
)

cat("\nRQ1 supplementary oral-reading c_nmt profile\n")
print(results, width = Inf)
cat("\nSaved ", output_stem, ".rds and _results.csv\n", sep = "")
