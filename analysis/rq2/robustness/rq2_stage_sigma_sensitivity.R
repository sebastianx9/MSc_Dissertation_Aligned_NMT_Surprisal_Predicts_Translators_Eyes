#!/usr/bin/env Rscript

# Targeted RQ2 residual-scale sensitivity. The mean and random-effects
# structure is identical to rq2_joint_maximal.R, but the Gaussian residual
# standard deviation is allowed to differ between reading and translation.

suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(posterior)
})
options(mc.cores = 4, warn = 1)

script_file <- sub(
  "^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
)
repo_root <- Sys.getenv("DISSERTATION_REPO_DIR", unset = "")
if (!nzchar(repo_root)) {
  repo_root <- normalizePath(
    file.path(dirname(script_file), "..", "..", ".."), mustWork = TRUE
  )
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
exclude_contrastive <- parse_bool(
  get_arg("--exclude-contrastive", "false"), "--exclude-contrastive"
)
dry_run <- parse_bool(get_arg("--dry-run", "false"), "--dry-run")
if (exclude_contrastive) {
  stop(
    "This targeted sensitivity is defined for the primary full sample only."
  )
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
path <- function(...) file.path(data_dir, ...)

fix_path <- path("fixation_durations_word.csv")
nmt_path <- path("nmt_surprisal_soft_word.csv")
mono_path <- path("monolingual_surprisal_word.csv")
freq_path <- path("subtlex_us.csv")
input_hashes <- analysis_input_hashes(c(
  fixation = fix_path,
  nmt_surprisal = nmt_path,
  monolingual_surprisal = mono_path,
  frequency = freq_path,
  analysis_design = file.path(repo_root, "analysis", "shared", "analysis_design.R"),
  analysis_script = file.path(
    repo_root, "analysis", "rq2", "robustness",
    "rq2_stage_sigma_sensitivity.R"
  )
))

fix <- read.csv(fix_path, stringsAsFactors = FALSE)
nmt <- read.csv(nmt_path, stringsAsFactors = FALSE)
mono <- read.csv(mono_path, stringsAsFactors = FALSE)
freq <- read.table(
  freq_path, sep = "\t", header = TRUE,
  stringsAsFactors = FALSE, quote = ""
) |>
  transmute(word_lower = lexical_form(Word), log10_freq = Lg10WF)

sentence_lengths <- nmt |>
  group_by(sentence_id) |>
  summarise(sentence_length = max(word_index) + 1, .groups = "drop")
predictors <- nmt |>
  left_join(sentence_lengths, by = "sentence_id") |>
  mutate(
    word_lower = lexical_form(word),
    word_length = nchar(word_lower, type = "chars"),
    word_position = word_index / (sentence_length - 1)
  ) |>
  left_join(freq, by = "word_lower") |>
  left_join(
    mono |>
      select(
        sentence_id, word_index,
        mono_surprisal = surprisal_sum
      ),
    by = c("sentence_id", "word_index")
  ) |>
  select(
    sentence_id, word_index, word_length, word_position,
    nmt_surprisal = surprisal_soft, mono_surprisal, log10_freq
  )

raw_data <- fix |>
  filter(stage %in% c("translate", "read")) |>
  left_join(predictors, by = c("sentence_id", "word_index")) |>
  mutate(
    log_tfd = log(total_fixation_duration_ms),
    ambiguity = factor(ambiguity),
    condition = as.integer(stage == "translate")
  ) |>
  filter(
    !is.na(nmt_surprisal), !is.na(mono_surprisal),
    !is.na(log10_freq)
  )
primary_reference <- raw_data |>
  anti_join(
    tibble(sentence_id = "S003", word_index = 3L),
    by = c("sentence_id", "word_index")
  )

z_from <- function(x, reference) {
  (x - mean(reference, na.rm = TRUE)) / sd(reference, na.rm = TRUE)
}
data <- primary_reference |>
  mutate(
    c_nmt = z_from(nmt_surprisal, primary_reference$nmt_surprisal),
    c_mono = z_from(mono_surprisal, primary_reference$mono_surprisal),
    c_wlen = z_from(word_length, primary_reference$word_length),
    c_wpos = z_from(word_position, primary_reference$word_position),
    c_freq = z_from(log10_freq, primary_reference$log10_freq)
  )

counts <- design_counts(data$sentence_id)
cat(sprintf(
  paste0(
    "Pooled N=%d (translate %d, read %d; sentence IDs=%d; ",
    "inference clusters=%d; stage-specific sigma)\n"
  ),
  nrow(data), sum(data$condition == 1), sum(data$condition == 0),
  counts[["n_sentence_ids"]], counts[["n_inference_clusters"]]
))

random_effects <- paste0(
  "(1 + condition + c_nmt + c_mono + condition:c_nmt + ",
  "condition:c_mono | participant) + ",
  "(1 + condition + c_nmt + c_mono + condition:c_nmt + ",
  "condition:c_mono | sentence_id)"
)
mean_formula <- as.formula(paste(
  "log_tfd ~ condition * (c_wlen + c_wpos + c_freq + ambiguity +",
  "c_nmt + c_mono) +", random_effects
))
formula <- bf(mean_formula, sigma ~ condition)

# Mean-model and random-effects priors match the primary joint model. A sigma
# regression uses a log link, so the primary exponential prior on one common
# sigma cannot be reused literally; weakly informative priors are instead set
# on the log residual scale and on the reading-to-translation log ratio.
priors <- c(
  prior(normal(0, 1), class = b),
  prior(normal(6, 1), class = Intercept),
  prior(exponential(1), class = sd),
  prior(lkj(2), class = cor),
  prior(normal(0, 1), class = Intercept, dpar = sigma),
  prior(normal(0, 0.5), class = b, dpar = sigma)
)

if (dry_run) {
  cat("Dry run: validating formula and prior targets without sampling.\n")
  print(get_prior(formula, data = data, family = gaussian()))
  invisible(make_stancode(
    formula, data = data, prior = priors, family = gaussian()
  ))
  quit(save = "no", status = 0)
}

model <- brm(
  formula, data = data, prior = priors,
  family = gaussian(),
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  chains = 4, iter = 8000, warmup = 4000,
  seed = 42, save_pars = save_pars(all = TRUE),
  silent = 2, refresh = 0
)
model <- set_analysis_input_hashes(model, input_hashes)
cache_path <- path("brm_cache", "rq2_joint_stage_sigma_v1.rds")
dir.create(dirname(cache_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(model, cache_path)

summarise_vector <- function(x) {
  c(
    Estimate = mean(x),
    Est.Error = sd(x),
    Q2.5 = unname(quantile(x, 0.025)),
    Q97.5 = unname(quantile(x, 0.975))
  )
}

draws <- as_draws_df(model)
required_draws <- c(
  "b_condition:c_nmt", "b_sigma_Intercept", "b_sigma_condition"
)
missing_draws <- setdiff(required_draws, names(draws))
if (length(missing_draws)) {
  stop("Missing expected posterior draws: ", paste(missing_draws, collapse = ", "))
}

sigma_read <- exp(draws$b_sigma_Intercept)
sigma_translate <- exp(
  draws$b_sigma_Intercept + draws$b_sigma_condition
)
sigma_ratio <- sigma_translate / sigma_read
sigma_output <- rbind(
  read = summarise_vector(sigma_read),
  translate = summarise_vector(sigma_translate),
  translate_over_read = summarise_vector(sigma_ratio)
)
sigma_output <- data.frame(
  quantity = rownames(sigma_output), sigma_output,
  row.names = NULL, check.names = FALSE
)
write.csv(
  sigma_output,
  file.path(output_dir, "rq2_stage_sigma_estimates.csv"),
  row.names = FALSE
)

stage_slopes <- hypothesis(
  model,
  c(
    nmt_read = "c_nmt = 0",
    nmt_translate = "c_nmt + condition:c_nmt = 0",
    mono_read = "c_mono = 0",
    mono_translate = "c_mono + condition:c_mono = 0"
  )
)
stage_slope_output <- data.frame(
  slope = rownames(stage_slopes$hypothesis),
  stage_slopes$hypothesis,
  row.names = NULL,
  check.names = FALSE
)
write.csv(
  stage_slope_output,
  file.path(output_dir, "rq2_stage_sigma_stage_slopes.csv"),
  row.names = FALSE
)

hetero_fixef <- fixef(model)
focal_term <- "condition:c_nmt"
if (!focal_term %in% rownames(hetero_fixef)) {
  stop("Missing focal coefficient: ", focal_term)
}
comparison <- data.frame(
  model = "stage-specific sigma",
  term = focal_term,
  Estimate = hetero_fixef[focal_term, "Estimate"],
  Est.Error = hetero_fixef[focal_term, "Est.Error"],
  Q2.5 = hetero_fixef[focal_term, "Q2.5"],
  Q97.5 = hetero_fixef[focal_term, "Q97.5"]
)
primary_path <- path("brm_cache", "rq2_joint_maximal_v4.rds")
if (file.exists(primary_path)) {
  primary <- readRDS(primary_path)
  primary_fixef <- fixef(primary)
  comparison <- bind_rows(
    data.frame(
      model = "common sigma (primary)",
      term = focal_term,
      Estimate = primary_fixef[focal_term, "Estimate"],
      Est.Error = primary_fixef[focal_term, "Est.Error"],
      Q2.5 = primary_fixef[focal_term, "Q2.5"],
      Q97.5 = primary_fixef[focal_term, "Q97.5"]
    ),
    comparison
  )
}
comparison$n_observations <- nrow(data)
comparison$n_sentence_ids <- counts[["n_sentence_ids"]]
comparison$n_inference_clusters <- counts[["n_inference_clusters"]]
write.csv(
  comparison,
  file.path(output_dir, "rq2_stage_sigma_focal_comparison.csv"),
  row.names = FALSE
)

divergences <- sum(
  subset(nuts_params(model), Parameter == "divergent__")$Value
)
cat("\nFocal comparison\n")
print(comparison, row.names = FALSE)
cat("\nResidual-scale estimates\n")
print(sigma_output, row.names = FALSE)
cat(sprintf(
  "\nRhat max: %.4f | divergences: %d\n",
  max(rhat(model), na.rm = TRUE), divergences
))
