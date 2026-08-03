#!/usr/bin/env Rscript

# RQ1 supplementary oral-reading profile: c_mono versus the full control
# baseline used throughout the dissertation. 

suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
})
options(mc.cores = 4, warn = 1)

ANALYSIS_VERSION <- "v3"
SEED <- 42L
K_FOLDS <- 10L
N_SIGN_FLIPS <- 10000L

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (!length(script_arg)) stop("Cannot determine the script path from --file.")
script_file <- sub("^--file=", "", script_arg[[1]])
repo_root <- Sys.getenv("DISSERTATION_REPO_DIR", unset = "")
if (!nzchar(repo_root)) {
  repo_root <- normalizePath(
    file.path(dirname(script_file), "..", "..", ".."), mustWork = TRUE
  )
}
helper_path <- file.path(repo_root, "analysis", "shared", "analysis_design.R")
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
exclude_contrastive <- parse_bool(
  get_arg("--exclude-contrastive", "false"),
  "--exclude-contrastive"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

fixation_path <- file.path(data_dir, "fixation_durations_word.csv")
nmt_path <- file.path(data_dir, "nmt_surprisal_soft_word.csv")
mono_path <- file.path(data_dir, "monolingual_surprisal_word.csv")
frequency_path <- file.path(data_dir, "subtlex_us.csv")
input_hashes <- analysis_input_hashes(c(
  fixation = fixation_path,
  nmt_surprisal = nmt_path,
  monolingual_surprisal = mono_path,
  frequency = frequency_path,
  analysis_design = helper_path,
  analysis_script = file.path(
    repo_root, "analysis", "rq1", "reading_profile", "rq1_reading_cmono.R"
  )
))

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

# Establish the full primary reading-stage complete-case sample first.  The
# leave-pair-out analysis below inherits this sample's z-scoring constants and
# fold labels rather than recomputing either after S031/S032 are removed.
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
if (any(!is.finite(primary_full$word_position))) {
  stop("Non-finite word positions remain after complete-case filtering.")
}

scale_variables <- c(
  "word_length", "word_position", "log10_freq", "mono_surprisal"
)
scaling <- tibble(
  variable = scale_variables,
  centre = vapply(
    primary_full[scale_variables], mean, numeric(1), na.rm = TRUE
  ),
  scale = vapply(
    primary_full[scale_variables], sd, numeric(1), na.rm = TRUE
  )
)
if (any(!is.finite(scaling$scale) | scaling$scale <= 0)) {
  stop("Every standardised predictor must have a finite, positive SD.")
}

z_from_primary <- function(x, variable) {
  row <- match(variable, scaling$variable)
  (x - scaling$centre[[row]]) / scaling$scale[[row]]
}
primary_full <- primary_full %>%
  mutate(
    c_wlen = z_from_primary(word_length, "word_length"),
    c_wpos = z_from_primary(word_position, "word_position"),
    c_freq = z_from_primary(log10_freq, "log10_freq"),
    c_mono = z_from_primary(mono_surprisal, "mono_surprisal")
  )

folds_full <- make_sentence_folds(
  primary_full$sentence_id, K = K_FOLDS, seed = SEED
)
keep <- contrastive_keep(
  primary_full$sentence_id, exclude_contrastive
)
data <- primary_full[keep, , drop = FALSE]
folds <- folds_full[keep]
sentence_id <- data$sentence_id

if (!nrow(data)) stop("No observations remain in the requested variant.")
stopifnot(
  length(folds) == nrow(data),
  all(tapply(
    folds, sentence_id,
    function(value) length(unique(value))
  ) == 1L)
)
assert_contrastive_fold_binding(folds, sentence_id)

counts <- design_counts(sentence_id)
N <- nrow(data)
J <- unname(counts[["n_sentence_ids"]])
G <- unname(counts[["n_inference_clusters"]])

cat(sprintf(
  paste0(
    "Reading stage: N=%d | sentence IDs=%d | inference clusters=%d | ",
    "participants=%d | contrastive pair=%s\n"
  ),
  N, J, G, n_distinct(data$participant),
  ifelse(exclude_contrastive, "excluded", "included and paired")
))
cat("Fold sizes (observations):\n")
print(table(folds))

random_effects <- "(1 | participant) + (1 | sentence_id)"
formulae <- list(
  base_with_position = as.formula(paste(
    "log_tfd ~ c_wlen + c_wpos + c_freq + ambiguity +",
    random_effects
  )),
  mono_with_position = as.formula(paste(
    "log_tfd ~ c_wlen + c_wpos + c_freq + ambiguity + c_mono +",
    random_effects
  ))
)
formula_text <- vapply(
  formulae, function(value) paste(deparse(value, width.cutoff = 500L),
                                  collapse = " "),
  character(1)
)

priors <- c(
  prior(normal(0, 1), class = b),
  prior(normal(6, 1), class = Intercept),
  prior(exponential(1), class = sd),
  prior(exponential(1), class = sigma)
)

cache_dir <- file.path(data_dir, "brm_cache")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

fit_kfold <- function(model_name) {
  cache_file <- variant_filename(
    sprintf(
      "rq1_reading_cmono_%s_%s.rds",
      ANALYSIS_VERSION, model_name
    ),
    exclude_contrastive
  )
  cache_path <- file.path(cache_dir, cache_file)
  expected_formula <- unname(formula_text[[model_name]])

  if (file.exists(cache_path)) {
    cat(sprintf("[%s] loading cached kfold\n", model_name))
    result <- readRDS(cache_path)
    assert_analysis_input_hashes(result, input_hashes, cache_path)
    stopifnot(
      inherits(result, "kfold"),
      length(result$pointwise[, "elpd_kfold"]) == N,
      identical(as.integer(attr(result, "folds")), as.integer(folds)),
      identical(attr(result, "analysis_version"), ANALYSIS_VERSION),
      identical(attr(result, "model_formula"), expected_formula)
    )
    return(result)
  }

  cat(sprintf("[%s] fitting model and 10-fold refits\n", model_name))
  started <- proc.time()
  model <- brm(
    formulae[[model_name]], data = data, prior = priors,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    chains = 4, iter = 4000, warmup = 2000,
    seed = SEED, silent = 2, refresh = 0
  )
  result <- kfold(
    model, folds = folds,
    chains = 4, iter = 4000, warmup = 2000,
    seed = SEED, silent = 2, refresh = 0
  )
  result <- set_analysis_input_hashes(result, input_hashes)
  attr(result, "folds") <- as.integer(folds)
  attr(result, "analysis_version") <- ANALYSIS_VERSION
  attr(result, "model_name") <- model_name
  attr(result, "model_formula") <- expected_formula
  attr(result, "exclude_contrastive") <- exclude_contrastive
  saveRDS(result, cache_path)
  cat(sprintf(
    "[%s] completed in %.1f minutes\n",
    model_name, (proc.time() - started)[["elapsed"]] / 60
  ))
  result
}

kfold_results <- lapply(names(formulae), fit_kfold)
names(kfold_results) <- names(formulae)
pointwise <- lapply(
  kfold_results,
  function(value) as.numeric(value$pointwise[, "elpd_kfold"])
)

sign_flip_one_sided <- function(cluster_delta, n_permutations = N_SIGN_FLIPS) {
  observed <- sum(cluster_delta)
  set.seed(SEED)
  permuted <- replicate(
    n_permutations,
    sum(
      cluster_delta * sample(
        c(-1, 1), length(cluster_delta), replace = TRUE
      )
    )
  )
  (1 + sum(permuted >= observed)) / (n_permutations + 1)
}

contrast_result <- function(name, interpretation, pointwise_delta) {
  cluster_delta <- cluster_delta_sums(pointwise_delta, sentence_id)
  stopifnot(length(cluster_delta) == G)
  elpd_diff <- sum(pointwise_delta)
  se_cluster <- sd(cluster_delta) * sqrt(G)
  se_pointwise <- sd(pointwise_delta) * sqrt(N)
  tibble(
    contrast = name,
    interpretation = interpretation,
    elpd_diff = elpd_diff,
    se_sentence_clustered = se_cluster,
    se_pointwise = se_pointwise,
    z_sentence_clustered = elpd_diff / se_cluster,
    p_signflip_one_sided = sign_flip_one_sided(cluster_delta),
    elpd_diff_per_observation = elpd_diff / N,
    n_observations = N,
    n_sentence_ids = J,
    n_inference_clusters = G
  )
}

gain_with_position <-
  pointwise$mono_with_position - pointwise$base_with_position

results <- bind_rows(
  contrast_result(
    "c_mono_with_position",
    "c_mono gain over lexical and positional controls",
    gain_with_position
  )
)

fold_assignment <- tibble(
  sentence_id = as.character(sentence_id),
  inference_cluster = contrastive_group_id(sentence_id),
  fold = as.integer(folds)
) %>%
  distinct() %>%
  arrange(fold, inference_cluster, sentence_id)

cluster_deltas <- list(
  c_mono_with_position = cluster_delta_sums(gain_with_position, sentence_id)
)

output <- list(
  metadata = list(
    analysis = "RQ1 supplementary oral-reading c_mono profile",
    analysis_version = ANALYSIS_VERSION,
    stage = "read",
    outcome = "log total fixation duration",
    seed = SEED,
    k_folds = K_FOLDS,
    fold_group = "sentence template; S031/S032 bound as one group",
    inference_cluster = "sentence template; S031/S032 bound as one cluster",
    n_sign_flips = N_SIGN_FLIPS,
    sign_flip_tail = "one-sided: reported elpd contrast greater than zero",
    sign_flip_correction = "(b + 1) / (B + 1)",
    scaling_reference = paste0(
      "primary full reading-stage complete-case sample before optional ",
      "S031/S032 exclusion"
    ),
    random_effects = random_effects,
    formulas = formula_text,
    lexical_form = paste0(
      "lowercase with surrounding Unicode punctuation removed; internal ",
      "apostrophes and hyphens retained"
    ),
    exclude_contrastive = exclude_contrastive,
    input_hashes = input_hashes
  ),
  sample = list(
    n_primary_reference = nrow(primary_full),
    n_observations = N,
    n_participants = n_distinct(data$participant),
    n_sentence_ids = J,
    n_inference_clusters = G
  ),
  scaling = scaling,
  fold_assignment = fold_assignment,
  results = results,
  pointwise_elpd = pointwise,
  pointwise_contrasts = list(
    c_mono_with_position = gain_with_position
  ),
  cluster_deltas = cluster_deltas,
  sentence_id = as.character(sentence_id)
)

output_file <- variant_filename(
  sprintf("rq1_reading_cmono_%s.rds", ANALYSIS_VERSION),
  exclude_contrastive
)
saveRDS(output, file.path(output_dir, output_file))

cat("\nRQ1 supplementary oral-reading c_mono profile\n")
print(results, width = Inf)
cat("\nSaved ", output_file, "\n", sep = "")
