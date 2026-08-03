#!/usr/bin/env Rscript

# Reading-stage joint-surprisal predictive comparison.
#
# The reading c_mono validation and c_nmt bridge already provide pointwise
# 10-fold elpd for:
#
#   M0      = lexical/positional controls
#   M_mono  = M0 + c_mono
#   M_nmt   = M0 + c_nmt
#
# This script verifies that those scores use the same observations and folds,
# then fits only:
#
#   M_both  = M0 + c_mono + c_nmt
#
# The resulting reciprocal additions test whether either surprisal contributes
# held-out information after the other is already present.  They complement
# the stand-alone comparison supplied by the reading-stage bridge; none of
# these within-reading contrasts substitutes for the pooled RQ2 stage
# interaction.

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
validation_script <- file.path(
  repo_root, "analysis", "rq1", "reading_profile", "rq1_reading_cmono.R"
)
bridge_script <- file.path(
  repo_root, "analysis", "rq1", "reading_profile", "rq1_reading_cnmt.R"
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

cache_dir <- get_arg("--cache-dir", file.path(data_dir, "brm_cache"))
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

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
bridge_name <- variant_filename(
  "rq1_reading_cnmt_v2.rds", exclude_contrastive
)
bridge_path <- normalizePath(
  get_arg(
    "--reading-bridge-rds",
    file.path(output_dir, bridge_name)
  ),
  mustWork = TRUE
)

fixation_path <- file.path(data_dir, "fixation_durations_word.csv")
nmt_path <- file.path(data_dir, "nmt_surprisal_soft_word.csv")
mono_path <- file.path(data_dir, "monolingual_surprisal_word.csv")
frequency_path <- file.path(data_dir, "subtlex_us.csv")

# Verify the two saved sources before reusing any pointwise predictions.
validation_hashes <- analysis_input_hashes(c(
  fixation = fixation_path,
  nmt_surprisal = nmt_path,
  monolingual_surprisal = mono_path,
  frequency = frequency_path,
  analysis_design = helper_path,
  analysis_script = validation_script
))
validation <- readRDS(validation_path)
stopifnot(
  identical(validation$metadata$analysis_version, "v3"),
  identical(validation$metadata$stage, "read"),
  identical(
    isTRUE(validation$metadata$exclude_contrastive),
    isTRUE(exclude_contrastive)
  )
)
if (!identical(validation$metadata$input_hashes, validation_hashes)) {
  stop(
    "The reading-validation output was created from different data or code; ",
    "do not reuse its pointwise scores."
  )
}

bridge_hashes <- analysis_input_hashes(c(
  fixation = fixation_path,
  nmt_surprisal = nmt_path,
  monolingual_surprisal = mono_path,
  frequency = frequency_path,
  reading_validation = validation_path,
  analysis_design = helper_path,
  analysis_script = bridge_script
))
bridge <- readRDS(bridge_path)
stopifnot(
  identical(bridge$metadata$analysis_version, "v2"),
  identical(bridge$metadata$stage, "read"),
  identical(bridge$metadata$seed, SEED),
  identical(bridge$metadata$k_folds, 10L),
  identical(
    bridge$metadata$nmt_formula,
    paste0(
      "log_tfd ~ c_wlen + c_wpos + c_freq + ambiguity + c_nmt + ",
      "(1 | participant) + (1 | sentence_id)"
    )
  )
)
bridge_common_names <- setdiff(names(bridge_hashes), "analysis_script")
if (!identical(
  bridge$metadata$input_hashes[bridge_common_names],
  bridge_hashes[bridge_common_names]
)) {
  stop(
    "The reading c_nmt bridge was created from different data or code; ",
    "do not reuse its pointwise scores."
  )
}
if (!identical(
  bridge$metadata$input_hashes[["analysis_script"]],
  bridge_hashes[["analysis_script"]]
)) {
  warning(
    "The bridge script hash has changed since the saved run. All data, ",
    "validation, design, formula, sample, fold, and pointwise checks must ",
    "therefore pass before its predictions are reused."
  )
}
bridge_variant <- unique(bridge$results$exclude_contrastive)
if (
  length(bridge_variant) != 1L ||
  !identical(isTRUE(bridge_variant), isTRUE(exclude_contrastive))
) {
  stop("The reading bridge and requested contrastive-pair variant differ.")
}

# Reconstruct the exact common analysis rows so M_both uses the same outcome,
# scaling, and observation order as the three saved models.
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

fold_lookup <- setNames(
  validation$fold_assignment$fold,
  validation$fold_assignment$sentence_id
)
folds <- as.integer(unname(fold_lookup[sentence_id]))
if (anyNA(folds)) stop("At least one sentence has no saved fold assignment.")
assert_contrastive_fold_binding(folds, sentence_id)

stopifnot(
  identical(sentence_id, validation$sentence_id),
  identical(sentence_id, as.character(bridge$sentence_id)),
  identical(N, validation$sample$n_observations),
  identical(N, bridge$sample$n_observations),
  identical(folds, as.integer(bridge$folds)),
  all(tapply(folds, sentence_id, function(x) length(unique(x))) == 1L)
)

pointwise <- list(
  base = as.numeric(bridge$pointwise_elpd$base_with_position),
  mono = as.numeric(bridge$pointwise_elpd$mono_with_position),
  nmt = as.numeric(bridge$pointwise_elpd$nmt_with_position)
)
if (!all(vapply(pointwise, length, integer(1)) == N)) {
  stop("Saved reading-stage pointwise elpd does not match the sample.")
}
stopifnot(
  identical(
    pointwise$base,
    as.numeric(validation$pointwise_elpd$base_with_position)
  ),
  identical(
    pointwise$mono,
    as.numeric(validation$pointwise_elpd$mono_with_position)
  ),
  isTRUE(all.equal(
    as.numeric(bridge$metadata$nmt_scaling),
    c(nmt_centre, nmt_scale),
    tolerance = 1e-12
  ))
)

cat(sprintf(
  paste0(
    "Reading joint sample: N=%d | participants=%d | sentence IDs=%d | ",
    "inference clusters=%d | contrastive pair=%s\n"
  ),
  N, n_distinct(data$participant), J, G,
  ifelse(exclude_contrastive, "excluded", "included and paired")
))
cat("Fold sizes:\n")
print(table(folds))

random_effects <- "(1 | participant) + (1 | sentence_id)"
both_formula <- as.formula(paste(
  "log_tfd ~ c_wlen + c_wpos + c_freq + ambiguity +",
  "c_mono + c_nmt +", random_effects
))
formula_text <- paste(
  deparse(both_formula, width.cutoff = 500L), collapse = " "
)

if (prepare_only) {
  cat("Preparation and three-model pointwise alignment checks passed.\n")
  cat("New model formula: ", formula_text, "\n", sep = "")
  quit(save = "no", status = 0L)
}

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
  reading_cnmt_bridge = bridge_path,
  analysis_design = helper_path,
  analysis_script = script_file
))

cache_name <- variant_filename(
  sprintf("rq1_reading_joint_surprisal_%s_both_kfold.rds",
          ANALYSIS_VERSION),
  exclude_contrastive
)
cache_path <- file.path(cache_dir, cache_name)

if (file.exists(cache_path)) {
  cat("Loading cached reading M_both k-fold result.\n")
  kfold_both <- readRDS(cache_path)
  assert_analysis_input_hashes(kfold_both, model_hashes, cache_path)
  stopifnot(
    inherits(kfold_both, "kfold"),
    length(kfold_both$pointwise[, "elpd_kfold"]) == N,
    identical(as.integer(attr(kfold_both, "folds")), folds),
    identical(attr(kfold_both, "model_formula"), formula_text)
  )
} else {
  cat("Fitting reading M_both and running grouped 10-fold CV...\n")
  started <- proc.time()
  model <- brm(
    both_formula, data = data, prior = priors,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    chains = 4, iter = 4000, warmup = 2000,
    seed = SEED, silent = 2, refresh = 0
  )
  kfold_both <- kfold(
    model, folds = folds,
    chains = 4, iter = 4000, warmup = 2000,
    seed = SEED, silent = 2, refresh = 0
  )
  kfold_both <- set_analysis_input_hashes(kfold_both, model_hashes)
  attr(kfold_both, "analysis_version") <- ANALYSIS_VERSION
  attr(kfold_both, "model_formula") <- formula_text
  attr(kfold_both, "exclude_contrastive") <- exclude_contrastive
  saveRDS(kfold_both, cache_path)
  cat(sprintf(
    "Reading M_both completed in %.1f minutes.\n",
    (proc.time() - started)[["elapsed"]] / 60
  ))
}

pointwise$both <- as.numeric(kfold_both$pointwise[, "elpd_kfold"])
if (length(pointwise$both) != N) {
  stop("M_both pointwise elpd does not match the common reading sample.")
}

sign_flip <- function(cluster_delta, tail, n = N_SIGN_FLIPS) {
  observed <- sum(cluster_delta)
  set.seed(SEED)
  permuted <- replicate(
    n,
    sum(cluster_delta * sample(c(-1, 1), length(cluster_delta), TRUE))
  )
  if (tail == "one-sided") {
    return((1 + sum(permuted >= observed)) / (n + 1))
  }
  if (tail == "two-sided") {
    return((1 + sum(abs(permuted) >= abs(observed))) / (n + 1))
  }
  stop("Unknown sign-flip tail: ", tail)
}

evaluate_contrast <- function(label, description, target, baseline, tail) {
  pointwise_delta <- pointwise[[target]] - pointwise[[baseline]]
  cluster_id <- contrastive_group_id(sentence_id)
  clusters <- tibble(
    contrast = label,
    sentence_id = sentence_id,
    cluster_id = cluster_id,
    fold = folds,
    delta_elpd = pointwise_delta
  ) %>%
    group_by(contrast, cluster_id) %>%
    summarise(
      sentence_ids = paste(sort(unique(sentence_id)), collapse = "+"),
      fold = first(fold),
      n_observations = n(),
      delta_elpd = sum(delta_elpd),
      .groups = "drop"
    ) %>%
    arrange(cluster_id)

  cluster_delta <- clusters$delta_elpd
  estimate <- sum(pointwise_delta)
  clustered_se <- sd(cluster_delta) * sqrt(length(cluster_delta))
  result <- tibble(
    contrast = label,
    interpretation = description,
    delta_elpd = estimate,
    sentence_clustered_se = clustered_se,
    ci_95_low = estimate - 1.96 * clustered_se,
    ci_95_high = estimate + 1.96 * clustered_se,
    p_signflip = sign_flip(cluster_delta, tail),
    tail = tail,
    p_signflip_holm_unique = NA_real_,
    per_word_delta = estimate / N,
    n_observations = N,
    n_sentence_ids = J,
    n_inference_clusters = length(cluster_delta),
    n_permutations = N_SIGN_FLIPS,
    seed = SEED,
    exclude_contrastive = exclude_contrastive
  )
  list(
    result = result,
    clusters = clusters,
    pointwise_delta = pointwise_delta
  )
}

comparisons <- list(
  evaluate_contrast(
    "M_nmt - M_controls",
    "Does c_nmt improve reading prediction beyond lexical/positional controls?",
    "nmt", "base", "one-sided"
  ),
  evaluate_contrast(
    "M_mono - M_controls",
    "Does c_mono improve reading prediction beyond lexical/positional controls?",
    "mono", "base", "one-sided"
  ),
  evaluate_contrast(
    "M_nmt - M_mono",
    "Which surprisal is the better stand-alone reading predictor?",
    "nmt", "mono", "two-sided"
  ),
  evaluate_contrast(
    "M_both - M_mono",
    "Does c_nmt add reading-stage information beyond c_mono?",
    "both", "mono", "one-sided"
  ),
  evaluate_contrast(
    "M_both - M_nmt",
    "Does c_mono add reading-stage information beyond c_nmt?",
    "both", "nmt", "one-sided"
  )
)

results <- bind_rows(lapply(comparisons, `[[`, "result"))
cluster_results <- bind_rows(lapply(comparisons, `[[`, "clusters"))

unique_rows <- results$contrast %in% c(
  "M_both - M_mono", "M_both - M_nmt"
)
results$p_signflip_holm_unique[unique_rows] <- p.adjust(
  results$p_signflip[unique_rows], method = "holm"
)

# On shared rows, the difference between the reciprocal additions must equal
# the direct stand-alone contrast, observation by observation.
direct <- comparisons[[3]]$pointwise_delta
increment_difference <-
  comparisons[[4]]$pointwise_delta - comparisons[[5]]$pointwise_delta
if (!isTRUE(all.equal(
  direct, increment_difference, tolerance = 1e-12
))) {
  stop("The direct and reciprocal-addition contrasts are not algebraically consistent.")
}

output_stem <- variant_filename(
  sprintf("rq1_reading_joint_surprisal_%s", ANALYSIS_VERSION),
  exclude_contrastive
)
results_path <- file.path(output_dir, paste0(output_stem, "_results.csv"))
clusters_path <- file.path(
  output_dir, paste0(output_stem, "_cluster_deltas.csv")
)
rds_path <- file.path(output_dir, paste0(output_stem, ".rds"))

write.csv(results, results_path, row.names = FALSE)
write.csv(cluster_results, clusters_path, row.names = FALSE)
saveRDS(
  list(
    metadata = list(
      analysis = "Reading-stage joint-surprisal predictive comparison",
      analysis_version = ANALYSIS_VERSION,
      stage = "read",
      outcome = "log total fixation duration",
      seed = SEED,
      k_folds = 10L,
      n_sign_flips = N_SIGN_FLIPS,
      random_effects = random_effects,
      both_formula = formula_text,
      validation_source = validation_path,
      bridge_source = bridge_path,
      input_hashes = model_hashes
    ),
    sample = list(
      n_observations = N,
      n_participants = n_distinct(data$participant),
      n_sentence_ids = J,
      n_inference_clusters = G
    ),
    results = results,
    cluster_deltas = cluster_results,
    pointwise_elpd = pointwise,
    pointwise_contrasts = setNames(
      lapply(comparisons, `[[`, "pointwise_delta"),
      results$contrast
    ),
    sentence_id = sentence_id,
    folds = folds
  ),
  rds_path
)

cat("\nReading-stage joint-surprisal predictive comparisons\n")
print(results, width = Inf)
cat(
  "\nSaved ", basename(rds_path), ", ", basename(results_path),
  ", and ", basename(clusters_path), "\n", sep = ""
)
