#!/usr/bin/env Rscript

# RQ1 translation-stage joint-surprisal comparisons.
# The three models already used by RQ1 provide two incremental contrasts:
#   controls + c_mono + c_nmt  versus  controls + c_mono;
#   controls + c_mono + c_nmt  versus  controls + c_nmt.
# They use the same complete-case sample, folds, priors, and cache names as
# rq1_kfold_elpd.R, so the single-predictor models are reused after RQ1 finishes.

suppressMessages({library(brms); library(dplyr)})
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
DATA_DIR <- normalizePath(
  get_arg("--data-dir", Sys.getenv("DISSERTATION_DATA_DIR", ".")),
  mustWork = TRUE
)
OUT <- get_arg(
  "--output-dir", Sys.getenv("DISSERTATION_OUTPUT_DIR", DATA_DIR)
)
exclude_contrastive <- parse_bool(
  get_arg("--exclude-contrastive", "false"), "--exclude-contrastive"
)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

fix_path <- file.path(DATA_DIR, "fixation_durations_word.csv")
nmt_path <- file.path(DATA_DIR, "nmt_surprisal_soft_word.csv")
mono_path <- file.path(DATA_DIR, "monolingual_surprisal_word.csv")
attn_path <- file.path(DATA_DIR, "attention_features_6_norm.csv")
freq_path <- file.path(DATA_DIR, "subtlex_us.csv")
input_hashes <- analysis_input_hashes(c(
  fixation=fix_path, nmt_surprisal=nmt_path,
  monolingual_surprisal=mono_path, attention_features=attn_path,
  frequency=freq_path,
  analysis_design=file.path(repo_root, "analysis", "shared", "analysis_design.R"),
  rq1_kfold_script=file.path(repo_root, "analysis", "rq1", "rq1_kfold_elpd.R"),
  rq1_joint_script=file.path(repo_root, "analysis", "rq1", "rq1_joint_surprisal_kfold.R")
))

fix <- read.csv(fix_path, stringsAsFactors = FALSE)
nmt <- read.csv(nmt_path, stringsAsFactors = FALSE)
mono <- read.csv(mono_path, stringsAsFactors = FALSE)
attn <- read.csv(attn_path, stringsAsFactors = FALSE)
freq <- read.table(
  freq_path, sep = "\t", header = TRUE,
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
  ) %>%
  select(
    sentence_id, word_index, word_length, word_position, log10_freq,
    nmt_surprisal = surprisal_soft, mono_surprisal,
    H_e, f_e, f_self, f_eos, f_recv, f_cross
  )

df <- fix %>%
  filter(stage == "translate") %>%
  left_join(predictors, by = c("sentence_id", "word_index")) %>%
  mutate(
    log_tfd = log(total_fixation_duration_ms),
    ambiguity = factor(ambiguity)
  ) %>%
  filter(
    !is.na(nmt_surprisal), !is.na(mono_surprisal), !is.na(log10_freq),
    !is.na(H_e), !is.na(f_e), !is.na(f_self), !is.na(f_eos), !is.na(f_recv),
    !is.na(f_cross)
  ) %>%
  anti_join(tibble(sentence_id = "S003", word_index = 3L),
            by = c("sentence_id", "word_index"))

z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
df <- df %>% mutate(
  c_nmt = z(nmt_surprisal), c_mono = z(mono_surprisal),
  c_wlen = z(word_length), c_wpos = z(word_position),
  c_freq = z(log10_freq)
)
fold_vec_full <- make_sentence_folds(df$sentence_id, K = 10, seed = 42)
keep <- contrastive_keep(df$sentence_id, exclude_contrastive)
df <- df[keep, , drop = FALSE]
fold_vec <- fold_vec_full[keep]
N <- nrow(df)
sid <- df$sentence_id
counts <- design_counts(sid)
J <- unname(counts[["n_sentence_ids"]])
G <- unname(counts[["n_inference_clusters"]])
cat(sprintf(
  "Translate: N=%d | sentence IDs=%d | inference clusters=%d | participants=%d | contrastive pair=%s\n",
  N, J, G, n_distinct(df$participant),
  ifelse(exclude_contrastive, "excluded", "included and paired")
))

stopifnot(
  N > 0L,
  all(tapply(fold_vec, sid, function(x) length(unique(x))) == 1L)
)
assert_contrastive_fold_binding(fold_vec, sid)

priors <- c(
  prior(normal(0, 1), class = b),
  prior(normal(6, 1), class = Intercept),
  prior(exponential(1), class = sd),
  prior(exponential(1), class = sigma)
)
CACHE <- file.path(DATA_DIR, "brm_cache")
dir.create(CACHE, showWarnings = FALSE)
CTRL <- "c_wlen + c_wpos + c_freq + ambiguity"
RE <- "(1 | participant) + (1 | sentence_id)"

fit_kfold <- function(name, formula) {
  cache_path <- file.path(
    CACHE,
    variant_filename(sprintf("rq1kf_v4_%s.rds", name),
                     exclude_contrastive)
  )
  if (file.exists(cache_path)) {
    cat(sprintf("[%s] loading cached kfold\n", name))
    result <- readRDS(cache_path)
    assert_analysis_input_hashes(result, input_hashes, cache_path)
    stopifnot(isTRUE(all.equal(
      as.numeric(attr(result, "folds")), as.numeric(fold_vec)
    )))
    return(result)
  }
  cat(sprintf("[%s] fitting + 10-fold refit\n", name))
  model <- brm(
    formula, data = df, prior = priors,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    chains = 4, iter = 4000, warmup = 2000, seed = 42,
    silent = 2, refresh = 0
  )
  result <- kfold(
    model, folds = fold_vec, chains = 4, iter = 4000, warmup = 2000,
    seed = 42, silent = 2, refresh = 0
  )
  result <- set_analysis_input_hashes(result, input_hashes)
  saveRDS(result, cache_path)
  result
}

mono_formula <- as.formula(paste(
  "log_tfd ~", CTRL, "+ c_mono +", RE
))
nmt_formula <- as.formula(paste(
  "log_tfd ~", CTRL, "+ c_nmt +", RE
))
both_formula <- as.formula(paste(
  "log_tfd ~", CTRL, "+ c_mono + c_nmt +", RE
))
pointwise <- list(
  mono = fit_kfold("c_mono", mono_formula)$pointwise[, "elpd_kfold"],
  nmt = fit_kfold("c_nmt", nmt_formula)$pointwise[, "elpd_kfold"],
  both = fit_kfold("c_mono_nmt", both_formula)$pointwise[, "elpd_kfold"]
)
stopifnot(all(vapply(pointwise, length, integer(1)) == N))

n_perm <- 10000L
seed <- 42L
evaluate_contrast <- function(label, target, baseline) {
  pointwise_delta <- pointwise[[target]] - pointwise[[baseline]]
  cluster_id <- contrastive_group_id(sid)
  cluster_results <- tibble(
    contrast = label,
    sentence_id = sid,
    cluster_id = cluster_id,
    fold = fold_vec,
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
  cluster_delta <- cluster_results$delta_elpd
  delta_elpd <- sum(pointwise_delta)
  clustered_se <- sd(cluster_delta) * sqrt(length(cluster_delta))
  set.seed(seed)
  permuted <- replicate(
    n_perm,
    sum(cluster_delta * sample(c(-1, 1), length(cluster_delta), TRUE))
  )
  result <- tibble(
    contrast = label,
    delta_elpd = delta_elpd,
    sentence_clustered_se = clustered_se,
    ci_95_low = delta_elpd - 1.96 * clustered_se,
    ci_95_high = delta_elpd + 1.96 * clustered_se,
    p_signflip_one_sided =
      (1 + sum(permuted >= delta_elpd)) / (n_perm + 1),
    p_signflip_two_sided =
      (1 + sum(abs(permuted) >= abs(delta_elpd))) / (n_perm + 1),
    per_word_delta = delta_elpd / N,
    n_observations = N,
    n_sentence_ids = J,
    n_inference_clusters = length(cluster_delta),
    n_permutations = n_perm,
    seed = seed,
    contrastive_pair = ifelse(exclude_contrastive, "excluded", "one cluster")
  )
  list(result = result, clusters = cluster_results,
       pointwise_delta = pointwise_delta)
}

comparisons <- list(
  evaluate_contrast("M_nmt - M_mono", "nmt", "mono"),
  evaluate_contrast("M_both - M_mono", "both", "mono"),
  evaluate_contrast("M_both - M_nmt", "both", "nmt")
)
results <- bind_rows(lapply(comparisons, `[[`, "result"))
cluster_results <- bind_rows(lapply(comparisons, `[[`, "clusters"))
results$p_signflip_one_sided_holm_unique <- NA_real_
unique_rows <- results$contrast %in% c(
  "M_both - M_mono", "M_both - M_nmt"
)
results$p_signflip_one_sided_holm_unique[unique_rows] <- p.adjust(
  results$p_signflip_one_sided[unique_rows], method = "holm"
)

results_name <- variant_filename(
  "rq1_joint_surprisal_results.csv", exclude_contrastive
)
cluster_name <- variant_filename(
  "rq1_joint_surprisal_cluster_deltas.csv", exclude_contrastive
)
write.csv(results, file.path(OUT, results_name), row.names = FALSE)
write.csv(cluster_results, file.path(OUT, cluster_name), row.names = FALSE)

# Preserve the established direct-comparison filenames for the dissertation
# tables while making the three-model result the authoritative source.
direct_result_name <- variant_filename(
  "rq1_direct_nmt_vs_mono_results.csv", exclude_contrastive
)
direct_delta_name <- variant_filename(
  "rq1_direct_nmt_vs_mono_sentence_deltas.csv", exclude_contrastive
)
write.csv(
  filter(results, contrast == "M_nmt - M_mono"),
  file.path(OUT, direct_result_name), row.names = FALSE
)
write.csv(
  filter(cluster_results, contrast == "M_nmt - M_mono") %>%
    select(-contrast),
  file.path(OUT, direct_delta_name), row.names = FALSE
)

output <- list(
  results = results,
  cluster_deltas = cluster_results,
  pointwise = pointwise,
  pointwise_contrasts = setNames(
    lapply(comparisons, `[[`, "pointwise_delta"),
    results$contrast
  ),
  sentence_id = sid,
  folds = fold_vec,
  N = N,
  J = J,
  G = G,
  exclude_contrastive = exclude_contrastive,
  input_hashes = input_hashes
)
saveRDS(
  output,
  file.path(OUT, variant_filename("rq1_joint_surprisal_kfold.rds",
                                  exclude_contrastive))
)

cat("\nRQ1 joint-surprisal predictive comparisons\n")
print(results, width = Inf)
cat("\nSaved ", results_name, " and ", cluster_name, "\n", sep = "")
