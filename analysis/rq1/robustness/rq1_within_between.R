#!/usr/bin/env Rscript

# RQ1 sensitivity: separate the source-word variation in c_nmt from its
# sentence-level component (Mundlak / within-between decomposition).
#
# The sentence mean is computed once over source positions, before participant
# rows are replicated.  The within-sentence component is each source word's
# deviation from that mean.  The sentence-level component cannot have a random
# slope by sentence because it is constant within a sentence.
suppressMessages({
  library(brms)
  library(dplyr)
})
options(mc.cores = 4, warn = 1)

script_file <- sub(
  "^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
)
repo_root <- Sys.getenv("DISSERTATION_REPO_DIR", unset = "")
if (!nzchar(repo_root)) {
  repo_root <- normalizePath(file.path(dirname(script_file), "..", "..", ".."), mustWork = TRUE)
}
source(file.path(repo_root, "analysis", "shared", "analysis_design.R"))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  hit <- grep(paste0("^", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", name, "="), "", hit[[1]])
}

DATA_DIR <- normalizePath(
  get_arg("--data-dir", Sys.getenv("DISSERTATION_DATA_DIR", ".")),
  mustWork = TRUE
)
OUT <- get_arg("--output-dir", Sys.getenv("DISSERTATION_OUTPUT_DIR", DATA_DIR))
exclude_contrastive <- parse_bool(
  get_arg("--exclude-contrastive", "false"), "--exclude-contrastive"
)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

fix_path <- file.path(DATA_DIR, "fixation_durations_word.csv")
nmt_path <- file.path(DATA_DIR, "nmt_surprisal_soft_word.csv")
freq_path <- file.path(DATA_DIR, "subtlex_us.csv")
input_hashes <- analysis_input_hashes(c(
  fixation = fix_path,
  nmt_surprisal = nmt_path,
  frequency = freq_path,
  analysis_design = file.path(repo_root, "analysis", "shared", "analysis_design.R"),
  analysis_script = file.path(
    repo_root, "analysis", "rq1", "robustness", "rq1_within_between.R"
  )
))

fix <- read.csv(fix_path, stringsAsFactors = FALSE)
nmt <- read.csv(nmt_path, stringsAsFactors = FALSE)
freq <- read.table(
  freq_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE, quote = ""
) %>%
  transmute(word_lower = lexical_form(Word), log10_freq = Lg10WF)

# Remove the identified copy-failure position before computing sentence means.
# All other source positions contribute to the sentence-level component,
# including positions with no fixation or missing lexical norms.
nmt_position <- nmt %>%
  anti_join(
    data.frame(sentence_id = "S003", word_index = 3L),
    by = c("sentence_id", "word_index")
  ) %>%
  group_by(sentence_id) %>%
  mutate(
    n_source_positions = n(),
    nmt_sentence_mean = mean(surprisal_soft),
    nmt_within = surprisal_soft - nmt_sentence_mean
  ) %>%
  ungroup()

# Numerical checks on the decomposition at the source-position level.
decomposition_check <- nmt_position %>%
  group_by(sentence_id) %>%
  summarise(
    n_source_positions = first(n_source_positions),
    nmt_sentence_mean = first(nmt_sentence_mean),
    nmt_sentence_total = sum(surprisal_soft),
    within_sum = sum(nmt_within),
    reconstruction_max_error = max(abs(
      surprisal_soft - (nmt_sentence_mean + nmt_within)
    )),
    .groups = "drop"
  )
stopifnot(
  max(abs(decomposition_check$within_sum)) < 1e-9,
  max(decomposition_check$reconstruction_max_error) < 1e-12
)

# Variance shares use unique source positions, so sentences are weighted by
# their number of retained words but repeated participant rows do not alter the
# decomposition.
raw_variance <- var(nmt_position$surprisal_soft)
decomposition_summary <- data.frame(
  n_source_positions = nrow(nmt_position),
  n_sentences = n_distinct(nmt_position$sentence_id),
  raw_variance = raw_variance,
  within_variance = var(nmt_position$nmt_within),
  between_variance = var(nmt_position$nmt_sentence_mean),
  within_variance_share = var(nmt_position$nmt_within) / raw_variance,
  between_variance_share = var(nmt_position$nmt_sentence_mean) / raw_variance,
  correlation_raw_within = cor(
    nmt_position$surprisal_soft, nmt_position$nmt_within
  ),
  correlation_raw_between = cor(
    nmt_position$surprisal_soft, nmt_position$nmt_sentence_mean
  )
)

sentence_length <- nmt %>%
  group_by(sentence_id) %>%
  summarise(sent_len = max(word_index) + 1L, .groups = "drop")

pred <- nmt_position %>%
  left_join(sentence_length, by = "sentence_id") %>%
  mutate(
    word_lower = lexical_form(word),
    word_length = nchar(word_lower, type = "chars"),
    word_position = word_index / (sent_len - 1)
  ) %>%
  left_join(freq, by = "word_lower") %>%
  select(
    sentence_id, word_index, word_length, word_position, log10_freq,
    nmt_surprisal = surprisal_soft, nmt_sentence_mean, nmt_within
  )

df <- fix %>%
  filter(stage == "translate") %>%
  left_join(pred, by = c("sentence_id", "word_index")) %>%
  mutate(
    log_tfd = log(total_fixation_duration_ms),
    ambiguity = factor(ambiguity)
  ) %>%
  filter(!is.na(nmt_surprisal), !is.na(log10_freq))

z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
df <- df %>%
  mutate(
    c_nmt_within = z(nmt_within),
    c_nmt_between = z(nmt_sentence_mean),
    c_wlen = z(word_length),
    c_wpos = z(word_position),
    c_freq = z(log10_freq)
  )
df <- apply_contrastive_sensitivity(df, exclude_contrastive)

counts <- design_counts(df$sentence_id)
cat(sprintf(
  paste0(
    "translate n=%d | sentence IDs=%d | inference clusters=%d | ",
    "contrastive pair=%s\n"
  ),
  nrow(df), counts[["n_sentence_ids"]], counts[["n_inference_clusters"]],
  ifelse(exclude_contrastive, "excluded", "included")
))
cat(sprintf(
  "corr(raw c_nmt, sentence mean)=%.4f | corr(raw c_nmt, within)=%.4f\n",
  cor(df$nmt_surprisal, df$nmt_sentence_mean),
  cor(df$nmt_surprisal, df$nmt_within)
))

priors <- c(
  prior(normal(0, 1), class = b),
  prior(normal(6, 1), class = Intercept),
  prior(exponential(1), class = sd),
  prior(exponential(1), class = sigma),
  prior(lkj(2), class = cor)
)

model <- brm(
  log_tfd ~ c_wlen + c_wpos + c_freq + ambiguity +
    c_nmt_within + c_nmt_between +
    (1 + c_nmt_within + c_nmt_between | participant) +
    (1 + c_nmt_within | sentence_id),
  data = df,
  prior = priors,
  control = list(adapt_delta = 0.99, max_treedepth = 14),
  chains = 4,
  iter = 4000,
  warmup = 2000,
  seed = 42,
  silent = 2,
  refresh = 0
)

dir.create(file.path(DATA_DIR, "brm_cache"), showWarnings = FALSE)
cache_name <- variant_filename(
  "rq1_within_between_v1.rds", exclude_contrastive
)
model <- set_analysis_input_hashes(model, input_hashes)
saveRDS(model, file.path(DATA_DIR, "brm_cache", cache_name))

fixed <- fixef(model)
result <- data.frame(
  term = rownames(fixed), fixed, row.names = NULL,
  n_observations = nrow(df),
  n_sentence_ids = counts[["n_sentence_ids"]],
  n_inference_clusters = counts[["n_inference_clusters"]],
  exclude_contrastive = exclude_contrastive
)
write.csv(
  result,
  file.path(
    OUT,
    variant_filename("rq1_within_between_results.csv", exclude_contrastive)
  ),
  row.names = FALSE
)
write.csv(
  decomposition_check,
  file.path(
    OUT,
    variant_filename(
      "rq1_within_between_sentence_components.csv", exclude_contrastive
    )
  ),
  row.names = FALSE
)
write.csv(
  decomposition_summary,
  file.path(
    OUT,
    variant_filename(
      "rq1_within_between_decomposition_summary.csv", exclude_contrastive
    )
  ),
  row.names = FALSE
)

cat("\n=== within-between fixed effects ===\n")
print(round(fixed, 4))
cat("\n=== random-effect SDs / correlations ===\n")
print(VarCorr(model))
divergences <- sum(subset(nuts_params(model), Parameter == "divergent__")$Value)
cat(sprintf(
  "\nRhat max: %.4f | divergences: %d | done\n",
  max(rhat(model), na.rm = TRUE), divergences
))
