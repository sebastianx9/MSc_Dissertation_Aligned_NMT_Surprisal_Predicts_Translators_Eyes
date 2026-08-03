#!/usr/bin/env Rscript

# RQ1 sensitivity checks:
#   (1) c_nmt beyond effective soft-alignment mass;
#   (2) c_nmt with the S003/stoplight observations restored.

suppressPackageStartupMessages({library(brms); library(dplyr)})
options(mc.cores = 4, warn = 1)

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])
repo_root <- Sys.getenv("DISSERTATION_REPO_DIR", unset = "")
if (!nzchar(repo_root)) {
  repo_root <- normalizePath(file.path(dirname(script_file), "..", "..", ".."), mustWork=TRUE)
}
source(file.path(repo_root, "analysis", "shared", "analysis_design.R"))

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
path <- function(...) file.path(data_dir, ...)

fix_path <- path("fixation_durations_word.csv")
nmt_path <- path("nmt_surprisal_soft_word.csv")
mono_path <- path("monolingual_surprisal_word.csv")
mass_path <- path("nmt_alignment_mass_word.csv")
attn_path <- path("attention_features_6_norm.csv")
freq_path <- path("subtlex_us.csv")
rq1_input_hashes <- analysis_input_hashes(c(
  fixation=fix_path, nmt_surprisal=nmt_path,
  monolingual_surprisal=mono_path, attention_features=attn_path,
  frequency=freq_path,
  analysis_design=file.path(repo_root, "analysis", "shared", "analysis_design.R"),
  rq1_kfold_script=file.path(repo_root, "analysis", "rq1", "rq1_kfold_elpd.R"),
  rq1_joint_script=file.path(repo_root, "analysis", "rq1", "rq1_joint_surprisal_kfold.R")
))
robustness_input_hashes <- analysis_input_hashes(c(
  fixation=fix_path, nmt_surprisal=nmt_path,
  monolingual_surprisal=mono_path, alignment_mass=mass_path,
  attention_features=attn_path, frequency=freq_path,
  analysis_design=file.path(repo_root, "analysis", "shared", "analysis_design.R"),
  robustness_script=file.path(
    repo_root, "analysis", "rq1", "robustness",
    "rq1_mass_stoplight_robustness.R"
  )
))

fix <- read.csv(fix_path, stringsAsFactors = FALSE)
nmt <- read.csv(nmt_path, stringsAsFactors = FALSE)
mono <- read.csv(mono_path, stringsAsFactors = FALSE)
mass <- read.csv(mass_path, stringsAsFactors = FALSE)
attn <- read.csv(attn_path, stringsAsFactors = FALSE)
freq <- read.table(freq_path, sep = "\t", header = TRUE,
                   stringsAsFactors = FALSE, quote = "") %>%
  transmute(word_lower = lexical_form(Word), log10_freq = Lg10WF)

# The mass extractor must reproduce the primary c_nmt values exactly at the
# stored six-decimal precision.
mass_check <- nmt %>%
  select(sentence_id, word_index, surprisal_soft) %>%
  inner_join(
    mass %>% select(sentence_id, word_index, surprisal_soft_recomputed),
    by = c("sentence_id", "word_index")
  )
stopifnot(
  nrow(mass_check) == nrow(nmt),
  max(abs(mass_check$surprisal_soft -
            mass_check$surprisal_soft_recomputed)) == 0
)

sentence_lengths <- nmt %>%
  group_by(sentence_id) %>%
  summarise(sentence_length = max(word_index) + 1, .groups = "drop")

predictors <- nmt %>%
  left_join(
    mass %>% select(sentence_id, word_index, alignment_mass,
                    surprisal_per_mass),
    by = c("sentence_id", "word_index")
  ) %>%
  left_join(sentence_lengths, by = "sentence_id") %>%
  mutate(
    word_lower = lexical_form(word),
    word_length = nchar(word_lower, type = "chars"),
    word_position = word_index / (sentence_length - 1)
  ) %>%
  left_join(freq, by = "word_lower") %>%
  left_join(
    mono %>% select(sentence_id, word_index,
                    mono_surprisal = surprisal_sum),
    by = c("sentence_id", "word_index")
  ) %>%
  left_join(
    attn %>% select(sentence_id, word_index,
                    H_e = attn_entropy, f_e = attn_context,
                    f_self = attn_self, f_eos = attn_eos,
                    f_recv = attn_recv,
                    f_cross = attn_cross),
    by = c("sentence_id", "word_index")
  ) %>%
  select(
    sentence_id, word_index, word_length, word_position, log10_freq,
    nmt_surprisal = surprisal_soft, mono_surprisal, alignment_mass,
    surprisal_per_mass, H_e, f_e, f_self, f_eos, f_recv, f_cross
  )

required_predictors <- c(
  "nmt_surprisal", "mono_surprisal", "alignment_mass",
  "surprisal_per_mass", "log10_freq", "H_e", "f_e", "f_self", "f_eos",
  "f_recv", "f_cross"
)
translate_all <- fix %>%
  filter(stage == "translate") %>%
  left_join(predictors, by = c("sentence_id", "word_index")) %>%
  mutate(log_tfd = log(total_fixation_duration_ms),
         ambiguity = factor(ambiguity)) %>%
  filter(if_all(all_of(required_predictors), ~ !is.na(.x)))
translate_clean <- translate_all %>%
  anti_join(tibble(sentence_id = "S003", word_index = 3L),
            by = c("sentence_id", "word_index"))

z_from <- function(x, reference) {
  (x - mean(reference, na.rm = TRUE)) / sd(reference, na.rm = TRUE)
}
scale_data <- function(data, reference) {
  data %>% mutate(
    c_nmt = z_from(nmt_surprisal, reference$nmt_surprisal),
    c_mass = z_from(alignment_mass, reference$alignment_mass),
    c_wlen = z_from(word_length, reference$word_length),
    c_wpos = z_from(word_position, reference$word_position),
    c_freq = z_from(log10_freq, reference$log10_freq),
    c_mono = z_from(mono_surprisal, reference$mono_surprisal)
  )
}
# Both robustness specifications use the primary, stoplight-excluded
# reference distribution. This keeps coefficient scales and priors identical
# when the excluded observations are restored.
scaling_reference <- translate_clean
translate_clean <- scale_data(translate_clean, scaling_reference)
translate_all <- scale_data(translate_all, scaling_reference)
stopifnot(
  nrow(translate_clean) > 0L,
  nrow(translate_all) >= nrow(translate_clean),
  nrow(translate_all) - nrow(translate_clean) ==
    sum(translate_all$sentence_id == "S003" &
          translate_all$word_index == 3L)
)

priors <- c(
  prior(normal(0, 1), class = b),
  prior(normal(6, 1), class = Intercept),
  prior(exponential(1), class = sd),
  prior(exponential(1), class = sigma)
)
cache_dir <- path("brm_cache")
dir.create(cache_dir, showWarnings = FALSE)
control_terms <- "c_wlen + c_wpos + c_freq + ambiguity"
random_intercepts <- "(1 | participant) + (1 | sentence_id)"
make_formula <- function(extra) {
  as.formula(paste("log_tfd ~", control_terms, extra, "+",
                   random_intercepts))
}

make_folds <- function(data) {
  folds <- make_sentence_folds(data$sentence_id, K = 10, seed = 42)
  stopifnot(all(tapply(folds, data$sentence_id,
                       function(x) length(unique(x))) == 1L))
  assert_contrastive_fold_binding(folds, data$sentence_id)
  folds
}

fit_kfold <- function(name, formula, data, folds) {
  cache_path <- file.path(cache_dir, paste0(name, "_v4.rds"))
  if (file.exists(cache_path)) {
    cached <- readRDS(cache_path)
    assert_analysis_input_hashes(
      cached, robustness_input_hashes, cache_path
    )
    stopifnot(
      length(cached$pointwise[, "elpd_kfold"]) == nrow(data),
      identical(as.integer(attr(cached, "folds")), as.integer(folds))
    )
    return(cached)
  }
  model <- brm(
    formula, data = data, prior = priors,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    chains = 4, iter = 4000, warmup = 2000, seed = 42,
    silent = 2, refresh = 0
  )
  result <- kfold(
    model, folds = folds, chains = 4, iter = 4000, warmup = 2000,
    seed = 42, silent = 2, refresh = 0
  )
  result <- set_analysis_input_hashes(result, robustness_input_hashes)
  saveRDS(result, cache_path)
  result
}

sign_flip <- function(sentence_delta, n_perm = 10000L) {
  observed <- sum(sentence_delta)
  set.seed(42)
  permuted <- replicate(
    n_perm,
    sum(sentence_delta * sample(c(-1, 1), length(sentence_delta), TRUE))
  )
  (1 + sum(permuted >= observed)) / (n_perm + 1)
}

compare_kfold <- function(target, baseline, sentence_id, contrast) {
  delta <- target$pointwise[, "elpd_kfold"] -
    baseline$pointwise[, "elpd_kfold"]
  sentence_delta <- cluster_delta_sums(delta, sentence_id)
  counts <- design_counts(sentence_id)
  tibble(
    contrast = contrast,
    delta_elpd = sum(delta),
    clustered_se = sqrt(length(sentence_delta)) * sd(sentence_delta),
    p_signflip_one_sided = sign_flip(sentence_delta),
    n_observations = length(delta),
    n_sentence_ids = counts[["n_sentence_ids"]],
    n_inference_clusters = length(sentence_delta)
  )
}

fold_clean <- make_folds(translate_clean)
primary_cache <- path("brm_cache", "rq1kf_v4_c_nmt.rds")
if (!file.exists(primary_cache)) {
  stop(
    "Missing primary RQ1 c_nmt cache despite the afterok dependency: ",
    primary_cache
  )
}
primary_kfold <- readRDS(primary_cache)
assert_analysis_input_hashes(primary_kfold, rq1_input_hashes,
                             primary_cache)
stopifnot(isTRUE(all.equal(
  as.numeric(fold_clean),
  as.numeric(attr(primary_kfold, "folds"))
)))

kf_mass <- fit_kfold(
  "rq1rob_mass_base", make_formula("+ c_mass"),
  translate_clean, fold_clean
)
kf_mass_nmt <- fit_kfold(
  "rq1rob_mass_cnmt", make_formula("+ c_mass + c_nmt"),
  translate_clean, fold_clean
)
result_mass <- compare_kfold(
  kf_mass_nmt, kf_mass, translate_clean$sentence_id,
  "c_nmt gain beyond alignment mass"
)

fold_all <- make_folds(translate_all)
kf_stop_base <- fit_kfold(
  "rq1rob_stoplight_base", make_formula(""), translate_all, fold_all
)
kf_stop_nmt <- fit_kfold(
  "rq1rob_stoplight_cnmt", make_formula("+ c_nmt"),
  translate_all, fold_all
)
result_stoplight <- compare_kfold(
  kf_stop_nmt, kf_stop_base, translate_all$sentence_id,
  "c_nmt gain with stoplight retained"
)

results <- bind_rows(result_mass, result_stoplight)
write.csv(results,
          file.path(output_dir, "rq1_mass_stoplight_predictive_results.csv"),
          row.names = FALSE)
print(results)

# Full-data coefficient checks use the same maximal random slopes as the
# corresponding RQ1 coefficient model.
fit_model <- function(name, formula, data) {
  cache_path <- file.path(cache_dir, paste0(name, "_v4.rds"))
  if (file.exists(cache_path)) {
    cached <- readRDS(cache_path)
    assert_analysis_input_hashes(
      cached, robustness_input_hashes, cache_path
    )
    return(cached)
  }
  model <- brm(
    formula, data = data, prior = c(priors, prior(lkj(2), class = cor)),
    control = list(adapt_delta = 0.99, max_treedepth = 14),
    chains = 4, iter = 4000, warmup = 2000, seed = 42,
    silent = 2, refresh = 0
  )
  model <- set_analysis_input_hashes(model, robustness_input_hashes)
  saveRDS(model, cache_path)
  model
}

mass_formula <- as.formula(paste(
  "log_tfd ~", control_terms, "+ c_mass + c_nmt +",
  "(1 + c_nmt | participant) + (1 + c_nmt | sentence_id)"
))
stop_formula <- as.formula(paste(
  "log_tfd ~", control_terms, "+ c_nmt +",
  "(1 + c_nmt | participant) + (1 + c_nmt | sentence_id)"
))
coef_mass <- fit_model(
  "rq1rob_coef_mass_cnmt", mass_formula, translate_clean
)
coef_stop <- fit_model(
  "rq1rob_coef_stoplight_cnmt", stop_formula, translate_all
)

extract_term <- function(model, term, analysis) {
  estimates <- fixef(model)
  tibble(
    analysis = analysis, term = term,
    estimate = estimates[term, "Estimate"],
    posterior_se = estimates[term, "Est.Error"],
    ci_low = estimates[term, "Q2.5"],
    ci_high = estimates[term, "Q97.5"]
  )
}
coefficient_results <- bind_rows(
  extract_term(coef_mass, "c_nmt", "c_nmt controlling alignment mass"),
  extract_term(coef_mass, "c_mass", "alignment mass controlling c_nmt"),
  extract_term(coef_stop, "c_nmt", "c_nmt with stoplight retained")
)
write.csv(
  coefficient_results,
  file.path(output_dir, "rq1_mass_stoplight_coefficient_results.csv"),
  row.names = FALSE
)
print(coefficient_results)
