#!/usr/bin/env Rscript

# ── RQ1 brms kfold elpd — 8 predictors vs baseline, TRANSLATE stage ───────────
# Bayesian replacement for the lmer 200-fold Table 3. Same convention as
# the RQ2 interaction check: sentence-grouped 10-fold (shared folds),
# per-model pointwise elpd, sentence-clustered SE (NOT loo_compare's se_diff),
# sentence-level sign-flip permutation. Within-translate z-scoring.
# ─────────────────────────────────────────────────────────────────────────────
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

fix  <- read.csv(fix_path,  stringsAsFactors=FALSE)
nmt  <- read.csv(nmt_path,  stringsAsFactors=FALSE)
mono <- read.csv(mono_path, stringsAsFactors=FALSE)
attn <- read.csv(attn_path, stringsAsFactors=FALSE)
freq <- read.table(freq_path, sep="\t",
                   header=TRUE, stringsAsFactors=FALSE, quote="") %>%
  transmute(word_lower=lexical_form(Word), log10_freq=Lg10WF)

sl <- nmt %>% group_by(sentence_id) %>% summarise(sent_len=max(word_index)+1, .groups="drop")
pred <- nmt %>% left_join(sl, by="sentence_id") %>%
  mutate(word_lower=lexical_form(word), word_length=nchar(word_lower, type="chars"),
         word_position=word_index/(sent_len-1)) %>%
  left_join(freq, by="word_lower") %>%
  left_join(mono %>% select(sentence_id, word_index, mono_surprisal=surprisal_sum),
            by=c("sentence_id","word_index")) %>%
  left_join(attn %>% select(sentence_id, word_index, H_e=attn_entropy,
                            f_e=attn_context, f_self=attn_self,
                            f_eos=attn_eos, f_recv=attn_recv,
                            f_cross=attn_cross),
            by=c("sentence_id","word_index")) %>%
  select(sentence_id, word_index, word_length, word_position, log10_freq,
         nmt_surprisal=surprisal_soft, mono_surprisal, H_e, f_e, f_self,
         f_eos, f_recv, f_cross)

df <- fix %>% filter(stage == "translate") %>%
  left_join(pred, by=c("sentence_id","word_index")) %>%
  mutate(log_tfd=log(total_fixation_duration_ms), ambiguity=factor(ambiguity)) %>%
  filter(!is.na(nmt_surprisal), !is.na(mono_surprisal), !is.na(log10_freq),
         !is.na(H_e), !is.na(f_e), !is.na(f_self), !is.na(f_eos),
         !is.na(f_recv), !is.na(f_cross)) %>%
  anti_join(data.frame(sentence_id="S003", word_index=3L), by=c("sentence_id","word_index"))

z <- function(x) (x-mean(x,na.rm=TRUE))/sd(x,na.rm=TRUE)
df <- df %>% mutate(c_nmt=z(nmt_surprisal), c_mono=z(mono_surprisal),
                    c_wlen=z(word_length), c_wpos=z(word_position), c_freq=z(log10_freq),
                    c_He=z(H_e), c_fe=z(f_e), c_fself=z(f_self),
                    c_feos=z(f_eos),
                    c_frecv=z(f_recv), c_fcross=z(f_cross))
# Apply leave-pair-out only after primary complete-case cleaning and scaling.
# Both variants therefore use the same full-sample centres, SDs, and fold
# labels for every retained sentence.
fold_vec_full <- make_sentence_folds(df$sentence_id, K=10, seed=42)
keep <- contrastive_keep(df$sentence_id, exclude_contrastive)
df <- df[keep, , drop=FALSE]
fold_vec <- fold_vec_full[keep]
N <- nrow(df); sid <- df$sentence_id
counts <- design_counts(sid)
J <- unname(counts[["n_sentence_ids"]])
G <- unname(counts[["n_inference_clusters"]])
cat(sprintf(
  "Translate: N=%d  sentence IDs=%d  inference clusters=%d  participants=%d  contrastive pair=%s\n\n",
  N, J, G, n_distinct(df$participant),
  ifelse(exclude_contrastive, "excluded", "included and paired")
))

assert_contrastive_fold_binding(fold_vec, sid)

priors <- c(prior(normal(0,1), class=b), prior(normal(6,1), class=Intercept),
            prior(exponential(1), class=sd), prior(exponential(1), class=sigma))
CACHE <- file.path(DATA_DIR, "brm_cache"); dir.create(CACHE, showWarnings=FALSE)
CTRL <- "c_wlen + c_wpos + c_freq + ambiguity"
RE   <- "(1|participant) + (1|sentence_id)"
f <- function(rhs) as.formula(paste("log_tfd ~", CTRL, rhs, "+", RE))

pred_vars <- c(
  "c_nmt", "c_mono", "c_He", "c_fe", "c_fself", "c_feos",
  "c_frecv", "c_fcross"
)
pred_labels <- c(
  "c_nmt", "c_mono", "H_e", "f_e", "f_self", "f_eos", "f_recv",
  "f_cross"
)

fit_kfold <- function(name, formula) {
  cache_name <- variant_filename(
    sprintf("rq1kf_v4_%s.rds", name), exclude_contrastive
  )
  path <- file.path(CACHE, cache_name)
  if (file.exists(path)) {
    cat(sprintf("[%s] cached\n", name))
    cached <- readRDS(path)
    assert_analysis_input_hashes(cached, input_hashes, path)
    stopifnot(
      length(cached$pointwise[, "elpd_kfold"]) == N,
      identical(as.integer(attr(cached, "folds")), as.integer(fold_vec))
    )
    return(cached)
  }
  cat(sprintf("[%s] fitting + 10-fold ...\n", name)); t0 <- proc.time()
  m  <- brm(formula, data=df, prior=priors, control=list(adapt_delta=0.95, max_treedepth=12),
            chains=4, iter=4000, warmup=2000, seed=42, silent=2, refresh=0)
  kf <- kfold(m, folds=fold_vec, chains=4, iter=4000, warmup=2000,
              seed=42, silent=2, refresh=0)
  kf <- set_analysis_input_hashes(kf, input_hashes)
  saveRDS(kf, path); cat(sprintf("[%s] %.0f min\n", name, (proc.time()-t0)["elapsed"]/60)); kf
}

ptw_base <- fit_kfold("base", f(""))$pointwise[, "elpd_kfold"]
sign_flip_p <- function(d_s, n=10000){ obs<-sum(d_s); set.seed(42)
  permuted <- replicate(n, sum(d_s*sample(c(-1,1),length(d_s),replace=TRUE)))
  (1 + sum(permuted >= obs)) / (n + 1) }

res <- data.frame(predictor=pred_labels, elpd_diff=NA, se_cluster=NA, se_naive=NA,
                  z=NA, p=NA, per_word=NA)
for (k in seq_along(pred_vars)) {
  ptw <- fit_kfold(pred_vars[k], f(paste("+", pred_vars[k])))$pointwise[, "elpd_kfold"]
  d_i <- ptw - ptw_base; d_s <- cluster_delta_sums(d_i, sid)
  res[k, -1] <- c(sum(d_i), sd(d_s)*sqrt(G), sd(d_i)*sqrt(N),
                  sum(d_i)/(sd(d_s)*sqrt(G)), sign_flip_p(d_s), sum(d_i)/N)
}
res$p_holm <- p.adjust(res$p, method="holm")

cat("\n══════════════════════════════════════════════════════════════════\n")
cat("RQ1 — brms sentence-grouped 10-fold elpd, sentence-clustered SE\n")
cat(sprintf("N=%d obs, J=%d sentence IDs, G=%d inference clusters\n", N, J, G))
cat("══════════════════════════════════════════════════════════════════\n")
print(format(res, digits=3, nsmall=3))
output_name <- variant_filename("rq1_kfold_elpd.rds", exclude_contrastive)
saveRDS(list(res=res, pointwise_base=ptw_base, sid=sid, N=N,
             J=J, G=G, S=G, folds=fold_vec,
             exclude_contrastive=exclude_contrastive,
             input_hashes=input_hashes),
        file.path(OUT, output_name))
cat("\nSaved ", output_name, "\n", sep="")
