#!/usr/bin/env Rscript

# RQ2 joint stage-interaction model. Every covariate is allowed to vary by
# stage, preventing stage structure in the controls from being absorbed by the
# surprisal interactions. Random slopes are maximal for the two surprisal
# predictors and their stage interactions.

suppressPackageStartupMessages({library(brms); library(dplyr)})
options(mc.cores = 4, warn = 1)

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])
repo_root <- Sys.getenv("DISSERTATION_REPO_DIR", unset = "")
if (!nzchar(repo_root)) {
  repo_root <- normalizePath(file.path(dirname(script_file), "..", ".."), mustWork=TRUE)
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
include_stoplight <- tolower(get_arg("--include-stoplight", "false")) == "true"
exclude_contrastive <- parse_bool(
  get_arg("--exclude-contrastive", "false"), "--exclude-contrastive"
)
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
path <- function(...) file.path(data_dir, ...)

fix_path <- path("fixation_durations_word.csv")
nmt_path <- path("nmt_surprisal_soft_word.csv")
mono_path <- path("monolingual_surprisal_word.csv")
freq_path <- path("subtlex_us.csv")
input_hashes <- analysis_input_hashes(c(
  fixation=fix_path, nmt_surprisal=nmt_path,
  monolingual_surprisal=mono_path, frequency=freq_path,
  analysis_design=file.path(repo_root, "analysis", "shared", "analysis_design.R"),
  analysis_script=file.path(repo_root, "analysis", "rq2", "rq2_joint_maximal.R")
))
fix <- read.csv(fix_path, stringsAsFactors = FALSE)
nmt <- read.csv(nmt_path, stringsAsFactors = FALSE)
mono <- read.csv(mono_path, stringsAsFactors = FALSE)
freq <- read.table(freq_path, sep = "\t", header = TRUE,
                   stringsAsFactors = FALSE, quote = "") %>%
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
    mono %>% select(sentence_id, word_index,
                    mono_surprisal = surprisal_sum),
    by = c("sentence_id", "word_index")
  ) %>%
  select(sentence_id, word_index, word_length, word_position,
         nmt_surprisal = surprisal_soft, mono_surprisal, log10_freq)

raw_data <- fix %>%
  filter(stage %in% c("translate", "read")) %>%
  left_join(predictors, by = c("sentence_id", "word_index")) %>%
  mutate(
    log_tfd = log(total_fixation_duration_ms),
    ambiguity = factor(ambiguity),
    condition = as.integer(stage == "translate")
  ) %>%
  filter(!is.na(nmt_surprisal), !is.na(mono_surprisal),
         !is.na(log10_freq))
primary_reference <- raw_data %>%
  anti_join(tibble(sentence_id = "S003", word_index = 3L),
            by = c("sentence_id", "word_index"))
data <- if (include_stoplight) raw_data else primary_reference

z_from <- function(x, reference) {
  (x - mean(reference, na.rm = TRUE)) / sd(reference, na.rm = TRUE)
}
data <- data %>% mutate(
  c_nmt = z_from(nmt_surprisal, primary_reference$nmt_surprisal),
  c_mono = z_from(mono_surprisal, primary_reference$mono_surprisal),
  c_wlen = z_from(word_length, primary_reference$word_length),
  c_wpos = z_from(word_position, primary_reference$word_position),
  c_freq = z_from(log10_freq, primary_reference$log10_freq)
)
data <- apply_contrastive_sensitivity(data, exclude_contrastive)
counts <- design_counts(data$sentence_id)
cat(sprintf(
  paste0("Pooled N=%d (translate %d, read %d; stoplight %s; ",
         "sentence IDs=%d; inference clusters=%d; contrastive pair %s)\n"),
  nrow(data), sum(data$condition == 1), sum(data$condition == 0),
  ifelse(include_stoplight, "included", "excluded"),
  counts[["n_sentence_ids"]], counts[["n_inference_clusters"]],
  ifelse(exclude_contrastive, "excluded", "included")
))

priors <- c(
  prior(normal(0, 1), class = b),
  prior(normal(6, 1), class = Intercept),
  prior(exponential(1), class = sd),
  prior(exponential(1), class = sigma),
  prior(lkj(2), class = cor)
)
random_effects <- paste0(
  "(1 + condition + c_nmt + c_mono + condition:c_nmt + ",
  "condition:c_mono | participant) + ",
  "(1 + condition + c_nmt + c_mono + condition:c_nmt + ",
  "condition:c_mono | sentence_id)"
)
formula <- as.formula(paste(
  "log_tfd ~ condition * (c_wlen + c_wpos + c_freq + ambiguity +",
  "c_nmt + c_mono) +", random_effects
))

model <- brm(
  formula, data = data, prior = priors,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  chains = 4, iter = 8000, warmup = 4000,
  seed = 42, save_pars = save_pars(all = TRUE), silent = 2, refresh = 0
)
dir.create(path("brm_cache"), showWarnings = FALSE)
cache_name <- if (include_stoplight) {
  "rq2rob_joint_stoplight_v3.rds"
} else {
  "rq2_joint_maximal_v4.rds"
}
cache_name <- variant_filename(cache_name, exclude_contrastive)
model <- set_analysis_input_hashes(model, input_hashes)
saveRDS(model, path("brm_cache", cache_name))

print(round(fixef(model), 4))
stage_slopes <- hypothesis(
  model,
  c(
    nmt_read = "c_nmt = 0",
    nmt_translate = "c_nmt + condition:c_nmt = 0",
    mono_read = "c_mono = 0",
    mono_translate = "c_mono + condition:c_mono = 0"
  )
)
print(stage_slopes)
stage_slope_output <- data.frame(
  slope=rownames(stage_slopes$hypothesis),
  stage_slopes$hypothesis,
  row.names=NULL,
  check.names=FALSE
)
write.csv(
  stage_slope_output,
  file.path(
    output_dir,
    variant_filename(
      if (include_stoplight) {
        "rq2_joint_stoplight_stage_slopes.csv"
      } else {
        "rq2_joint_stage_slopes.csv"
      },
      exclude_contrastive
    )
  ),
  row.names=FALSE
)
divergences <- sum(
  subset(nuts_params(model), Parameter == "divergent__")$Value
)
cat(sprintf(
  "Rhat max: %.4f | divergences: %d\n",
  max(rhat(model), na.rm = TRUE), divergences
))

coefficient_output <- data.frame(
  term=rownames(fixef(model)), fixef(model), row.names=NULL,
  n_observations=nrow(data),
  n_sentence_ids=counts[["n_sentence_ids"]],
  n_inference_clusters=counts[["n_inference_clusters"]],
  exclude_contrastive=exclude_contrastive,
  include_stoplight=include_stoplight
)
base_output <- if (include_stoplight) {
  "rq2_joint_stoplight_results.csv"
} else {
  "rq2_joint_maximal_results.csv"
}
write.csv(
  coefficient_output,
  file.path(output_dir,
            variant_filename(base_output, exclude_contrastive)),
  row.names=FALSE
)
