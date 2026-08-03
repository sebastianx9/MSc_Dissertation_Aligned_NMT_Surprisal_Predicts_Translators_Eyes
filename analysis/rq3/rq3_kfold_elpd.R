#!/usr/bin/env Rscript

# RQ3 phase profile on FFD, GD, go-past, and conditional RRT.
#
# The focal predictive contrast locates the overall c_nmt association:
#   controls + c_nmt  versus  controls.
# Full-data c_nmt coefficient models with focal random slopes provide direction
# and effect-size estimates; an ELPD gain alone does not identify the sign of
# the association. Comparisons conditional on c_mono belong to RQ1/RQ2 rather
# than this phase-localisation analysis.
# Outcome-specific coefficients, credible intervals, and clustered ELPD
# intervals are interpreted jointly as a phase profile. RQ3 does not use
# sign-flip p-values or a multiplicity adjustment.
suppressMessages({library(brms); library(dplyr)}); options(mc.cores = 4, warn = 1)

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
OUT <- get_arg("--output-dir", Sys.getenv("DISSERTATION_OUTPUT_DIR", DATA_DIR))
exclude_contrastive <- parse_bool(
  get_arg("--exclude-contrastive", "false"), "--exclude-contrastive"
)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

eye_path <- file.path(DATA_DIR, "eye_measures_word.csv")
nmt_path <- file.path(DATA_DIR, "nmt_surprisal_soft_word.csv")
freq_path <- file.path(DATA_DIR, "subtlex_us.csv")
input_hashes <- analysis_input_hashes(c(
  eye_measures=eye_path, nmt_surprisal=nmt_path,
  frequency=freq_path,
  analysis_design=file.path(repo_root, "analysis", "shared", "analysis_design.R"),
  analysis_script=file.path(repo_root, "analysis", "rq3", "rq3_kfold_elpd.R")
))

em   <- read.csv(eye_path,  stringsAsFactors=FALSE)
nmt  <- read.csv(nmt_path,  stringsAsFactors=FALSE)
freq_raw <- read.table(
  freq_path, sep="\t", header=TRUE, stringsAsFactors=FALSE, quote=""
)
required_eye_columns <- c(
  "participant", "sentence_id", "stage", "word_index",
  "ffd_ms", "gd_ms", "go_past_ms", "rrt_ms", "reread_occurrence",
  "first_encounter_status", "ambiguity"
)
missing_eye_columns <- setdiff(required_eye_columns, names(em))
if (length(missing_eye_columns)) {
  stop(
    "eye_measures_word.csv must be regenerated with the current extractor; ",
    "missing columns: ", paste(missing_eye_columns, collapse=", ")
  )
}
required_nmt_columns <- c(
  "sentence_id", "word_index", "word", "surprisal_soft"
)
missing_nmt_columns <- setdiff(required_nmt_columns, names(nmt))
if (length(missing_nmt_columns)) {
  stop(
    "nmt_surprisal_soft_word.csv is missing columns: ",
    paste(missing_nmt_columns, collapse=", ")
  )
}
required_frequency_columns <- c("Word", "Lg10WF")
missing_frequency_columns <- setdiff(required_frequency_columns, names(freq_raw))
if (length(missing_frequency_columns)) {
  stop(
    "subtlex_us.csv is missing columns: ",
    paste(missing_frequency_columns, collapse=", ")
  )
}
freq <- freq_raw %>%
  transmute(word_lower=lexical_form(Word), log10_freq=Lg10WF)
sl <- nmt %>% group_by(sentence_id) %>% summarise(sent_len=max(word_index)+1,.groups="drop")
pred <- nmt %>% left_join(sl,by="sentence_id") %>%
  mutate(word_lower=lexical_form(word),word_length=nchar(word_lower, type="chars"),
         word_position=word_index/(sent_len-1)) %>%
  left_join(freq,by="word_lower") %>%
  select(sentence_id,word_index,word_length,word_position,log10_freq,
         nmt_surprisal=surprisal_soft)
base_df <- em %>% filter(stage=="translate") %>% left_join(pred,by=c("sentence_id","word_index")) %>%
  mutate(ambiguity=factor(ambiguity)) %>%
  filter(!is.na(nmt_surprisal),!is.na(log10_freq)) %>%
  anti_join(data.frame(sentence_id="S003",word_index=3L),by=c("sentence_id","word_index"))
z<-function(x)(x-mean(x,na.rm=TRUE))/sd(x,na.rm=TRUE)
base_df <- base_df %>% mutate(c_nmt=z(nmt_surprisal),
                              c_wlen=z(word_length),c_wpos=z(word_position),
                              c_freq=z(log10_freq))
if (any(
  !is.na(base_df$go_past_ms) &
    (!is.finite(base_df$go_past_ms) | base_df$go_past_ms <= 0)
)) {
  stop("Observed go_past_ms values must be finite and positive.")
}
if (any(
  !is.na(base_df$go_past_ms) & !is.na(base_df$gd_ms) &
    base_df$go_past_ms + 0.01 < base_df$gd_ms
)) {
  stop("Observed go_past_ms cannot be shorter than gd_ms.")
}
df_ffd <- base_df %>% filter(!is.na(ffd_ms), ffd_ms > 0) %>% mutate(log_ffd = log(ffd_ms))
df_gd  <- base_df %>% filter(!is.na(gd_ms),  gd_ms  > 0) %>% mutate(log_gd  = log(gd_ms))
df_go_past <- base_df %>%
  filter(!is.na(go_past_ms), go_past_ms > 0) %>%
  mutate(log_go_past = log(go_past_ms))
df_rrt <- base_df %>%
  filter(reread_occurrence == 1, rrt_ms > 0) %>%
  mutate(log_rrt = log(rrt_ms))
cat(sprintf("FFD n=%d  GD n=%d  go-past n=%d  conditional RRT n=%d\n",
            nrow(df_ffd), nrow(df_gd), nrow(df_go_past), nrow(df_rrt)))

pri<-c(prior(normal(0,1),class=b),prior(normal(6,1),class=Intercept),prior(exponential(1),class=sd),prior(exponential(1),class=sigma))
coef_pri <- c(pri, prior(lkj(2), class=cor))
CACHE<-file.path(DATA_DIR,"brm_cache"); dir.create(CACHE, showWarnings=FALSE)
CTRL<-"c_wlen+c_wpos+c_freq+ambiguity"; RE<-"(1|participant)+(1|sentence_id)"

run_outcome <- function(df, y, tag) {
  df <- df[!is.na(df[[y]]), ]
  fv_full <- make_sentence_folds(df$sentence_id, K=10, seed=42)
  keep <- contrastive_keep(df$sentence_id, exclude_contrastive)
  df <- df[keep, , drop=FALSE]
  fv <- fv_full[keep]
  N<-nrow(df); sid<-df$sentence_id
  counts <- design_counts(sid)
  J <- unname(counts[["n_sentence_ids"]])
  G <- unname(counts[["n_inference_clusters"]])
  assert_contrastive_fold_binding(fv, sid)
  f<-function(rhs) as.formula(paste(y,"~",CTRL,rhs,"+",RE))
  fitkf<-function(nm,form){p<-file.path(
      CACHE,
      variant_filename(sprintf("rq3kf_v6_%s_%s.rds",tag,nm),
                       exclude_contrastive)
    ); if(file.exists(p)){
      cached <- readRDS(p)
      assert_analysis_input_hashes(cached, input_hashes, p)
      stopifnot(
        length(cached$pointwise[,"elpd_kfold"]) == N,
        identical(as.integer(attr(cached,"folds")), as.integer(fv))
      )
      return(cached)
    }
    m<-brm(form,data=df,prior=pri,control=list(adapt_delta=0.95,max_treedepth=12),chains=4,iter=4000,warmup=2000,seed=42,silent=2,refresh=0)
    kf<-kfold(m,folds=fv,chains=4,iter=4000,warmup=2000,seed=42,silent=2,refresh=0)
    kf<-set_analysis_input_hashes(kf,input_hashes)
    saveRDS(kf,p);kf}
  pointwise <- list(
    base = fitkf("base", f(""))$pointwise[,"elpd_kfold"],
    c_nmt = fitkf("c_nmt", f("+ c_nmt"))$pointwise[,"elpd_kfold"]
  )
  compare <- function(contrast, target, baseline, role) {
    di <- pointwise[[target]] - pointwise[[baseline]]
    ds <- cluster_delta_sums(di, sid)
    data.frame(
      outcome=tag, contrast=contrast,
      focal_predictor=sub("_(unique|total)$", "", contrast),
      target_model=target, baseline_model=baseline, role=role,
      elpd_diff=sum(di), se_cluster=sd(ds)*sqrt(G),
      ci_95_low=sum(di)-1.96*sd(ds)*sqrt(G),
      ci_95_high=sum(di)+1.96*sd(ds)*sqrt(G),
      n_observations=N, n_sentence_ids=J, n_inference_clusters=G,
      fold_seed=42L,
      stringsAsFactors=FALSE
    )
  }
  predictive <- compare("c_nmt_total", "c_nmt", "base", "focal")

  coefficient_formula <- as.formula(paste(
    y, "~", CTRL, "+ c_nmt +",
    "(1 + c_nmt | participant) + (1 + c_nmt | sentence_id)"
  ))
  coefficient_path <- file.path(
    CACHE,
    variant_filename(sprintf("rq3coef_v3_total_%s.rds", tag),
                     exclude_contrastive)
  )
  if (file.exists(coefficient_path)) {
    coefficient_model <- readRDS(coefficient_path)
    assert_analysis_input_hashes(
      coefficient_model, input_hashes, coefficient_path
    )
    stopifnot(nrow(coefficient_model$data) == N)
  } else {
    coefficient_model <- brm(
      coefficient_formula, data=df, prior=coef_pri,
      control=list(adapt_delta=0.99,max_treedepth=14),
      chains=4,iter=4000,warmup=2000,seed=42,silent=2,refresh=0
    )
    coefficient_model <- set_analysis_input_hashes(
      coefficient_model, input_hashes
    )
    saveRDS(coefficient_model, coefficient_path)
  }
  fixed <- fixef(coefficient_model)
  keep_terms <- intersect("c_nmt", rownames(fixed))
  stopifnot(identical(keep_terms, "c_nmt"))
  parameter_names <- paste0("b_", keep_terms)
  fixed_diagnostics <- posterior::summarise_draws(
    posterior::as_draws_array(
      coefficient_model, variable=parameter_names
    ),
    "rhat", "ess_bulk", "ess_tail"
  )
  fixed_diagnostics <- as.data.frame(fixed_diagnostics)
  stopifnot(all(parameter_names %in% fixed_diagnostics$variable))
  fixed_diagnostics <- fixed_diagnostics[
    match(parameter_names, fixed_diagnostics$variable),,
    drop=FALSE
  ]
  sampler_parameters <- nuts_params(coefficient_model)
  divergences <- sum(
    subset(sampler_parameters, Parameter=="divergent__")$Value
  )
  treedepth_hits <- sum(
    subset(sampler_parameters, Parameter=="treedepth__")$Value >= 14
  )
  energy <- subset(sampler_parameters, Parameter=="energy__")
  bfmi_by_chain <- tapply(
    energy$Value, energy$Chain,
    function(values) mean(diff(values)^2) / stats::var(values)
  )
  coefficients <- data.frame(
    outcome=tag, term=keep_terms, fixed[keep_terms,,drop=FALSE],
    Rhat=fixed_diagnostics$rhat,
    Bulk_ESS=fixed_diagnostics$ess_bulk,
    Tail_ESS=fixed_diagnostics$ess_tail,
    n_observations=N, n_sentence_ids=J, n_inference_clusters=G,
    max_rhat=max(rhat(coefficient_model),na.rm=TRUE),
    divergences=divergences, treedepth_hits=treedepth_hits,
    min_bfmi=min(bfmi_by_chain, na.rm=TRUE),
    row.names=NULL, check.names=FALSE
  )
  list(predictive=predictive, coefficients=coefficients,
       pointwise=pointwise, folds=fv, sentence_id=sid)
}
outcomes <- list(
  FFD=run_outcome(df_ffd, "log_ffd", "FFD"),
  GD=run_outcome(df_gd, "log_gd", "GD"),
  Go_past=run_outcome(df_go_past, "log_go_past", "Go-past"),
  RRT=run_outcome(df_rrt, "log_rrt", "RRT")
)
res <- bind_rows(lapply(outcomes, `[[`, "predictive"))
coefficient_results <- bind_rows(lapply(outcomes, `[[`, "coefficients"))
res$exclude_contrastive <- exclude_contrastive
coefficient_results$exclude_contrastive <- exclude_contrastive
attr(res, "outcome_definitions") <- c(
  FFD="first fixation duration on a word's first encounter",
  GD="gaze duration during a word's first encounter",
  `Go-past`=paste(
    "go-past fixation time from first landing through the fixation before",
    "the first subsequent rightward crossing; structurally undefined without",
    "a crossing or when first encountered by regression-in"
  ),
  RRT="re-reading duration conditional on at least one post-first-encounter fixation"
)
output <- list(
  predictive=res,
  coefficients=coefficient_results,
  pointwise=lapply(outcomes, `[[`, "pointwise"),
  folds=lapply(outcomes, `[[`, "folds"),
  sentence_id=lapply(outcomes, `[[`, "sentence_id"),
  exclude_contrastive=exclude_contrastive,
  inferential_scope="phase_profile_no_permutation",
  input_hashes=input_hashes,
  outcome_definitions=attr(res, "outcome_definitions")
)
output <- set_analysis_input_hashes(output, input_hashes)
saveRDS(output, file.path(
  OUT, variant_filename("rq3_kfold_elpd.rds", exclude_contrastive)
))
write.csv(
  res,
  file.path(OUT, variant_filename("rq3_kfold_elpd_results.csv",
                                  exclude_contrastive)),
  row.names=FALSE
)
write.csv(
  coefficient_results,
  file.path(OUT, variant_filename("rq3_coefficient_results.csv",
                                  exclude_contrastive)),
  row.names=FALSE
)
cat("=== RQ3 phase-localising c_nmt predictive contrasts ===\n")
print(format(res, digits=3))
cat("\n=== RQ3 c_nmt coefficient estimates ===\n")
print(format(coefficient_results, digits=3))
cat("DONE\n")
