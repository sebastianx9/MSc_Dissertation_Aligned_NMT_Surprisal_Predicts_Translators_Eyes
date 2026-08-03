#!/usr/bin/env Rscript

# RQ1 c_nmt coefficient, random slopes maximal for the focal predictor.
# RE: (1 + c_nmt | participant) + (1 + c_nmt | sentence_id).
# Reports convergence diagnostics + RE SDs so degeneracy is visible.
suppressMessages({library(brms); library(dplyr)}); options(mc.cores=4, warn=1)

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
dir.create(OUT, recursive=TRUE, showWarnings=FALSE)
fix_path <- file.path(DATA_DIR, "fixation_durations_word.csv")
nmt_path <- file.path(DATA_DIR, "nmt_surprisal_soft_word.csv")
freq_path <- file.path(DATA_DIR, "subtlex_us.csv")
input_hashes <- analysis_input_hashes(c(
  fixation=fix_path, nmt_surprisal=nmt_path, frequency=freq_path,
  analysis_design=file.path(repo_root, "analysis", "shared", "analysis_design.R"),
  analysis_script=file.path(repo_root, "analysis", "rq1", "rq1_coef_maximal.R")
))
fix  <- read.csv(fix_path, stringsAsFactors=FALSE)
nmt  <- read.csv(nmt_path, stringsAsFactors=FALSE)
freq <- read.table(freq_path, sep="\t", header=TRUE,
                   stringsAsFactors=FALSE, quote="") %>%
  transmute(word_lower=lexical_form(Word), log10_freq=Lg10WF)
sl <- nmt %>% group_by(sentence_id) %>% summarise(sent_len=max(word_index)+1,.groups="drop")
pred <- nmt %>% left_join(sl,by="sentence_id") %>%
  mutate(word_lower=lexical_form(word),word_length=nchar(word_lower, type="chars"),
         word_position=word_index/(sent_len-1)) %>%
  left_join(freq,by="word_lower") %>%
  select(sentence_id,word_index,word_length,word_position,nmt_surprisal=surprisal_soft,log10_freq)
df <- fix %>% filter(stage=="translate") %>% left_join(pred,by=c("sentence_id","word_index")) %>%
  mutate(log_tfd=log(total_fixation_duration_ms),ambiguity=factor(ambiguity)) %>%
  filter(!is.na(nmt_surprisal),!is.na(log10_freq)) %>%
  anti_join(data.frame(sentence_id="S003",word_index=3L),by=c("sentence_id","word_index"))
z<-function(x)(x-mean(x,na.rm=TRUE))/sd(x,na.rm=TRUE)
df<-df%>%mutate(c_nmt=z(nmt_surprisal),c_wlen=z(word_length),c_wpos=z(word_position),c_freq=z(log10_freq))
df <- apply_contrastive_sensitivity(df, exclude_contrastive)
counts <- design_counts(df$sentence_id)
cat(sprintf(
  "translate n=%d | sentence IDs=%d | inference clusters=%d | contrastive pair=%s\n",
  nrow(df), counts[["n_sentence_ids"]], counts[["n_inference_clusters"]],
  ifelse(exclude_contrastive, "excluded", "included")
))
pri<-c(prior(normal(0,1),class=b),prior(normal(6,1),class=Intercept),
       prior(exponential(1),class=sd),prior(exponential(1),class=sigma),prior(lkj(2),class=cor))
m<-brm(log_tfd~c_wlen+c_wpos+c_freq+ambiguity+c_nmt+
         (1+c_nmt|participant)+(1+c_nmt|sentence_id),
       data=df,prior=pri,control=list(adapt_delta=0.99,max_treedepth=14),
       chains=4,iter=4000,warmup=2000,seed=42,silent=2,refresh=0)
dir.create(file.path(DATA_DIR,"brm_cache"), showWarnings=FALSE)
cache_name <- variant_filename("rq1_coef_maximal_v2.rds", exclude_contrastive)
m <- set_analysis_input_hashes(m, input_hashes)
saveRDS(m,file.path(DATA_DIR,"brm_cache",cache_name))
cat("\n=== fixef (MAXIMAL RE) ===\n"); print(round(fixef(m),4))
cat("\n=== RE SDs / correlations ===\n"); print(VarCorr(m))
nd<-sum(subset(nuts_params(m),Parameter=="divergent__")$Value)
cat(sprintf("\nRhat max: %.4f | divergences: %d | done\n", round(max(rhat(m),na.rm=TRUE),4), nd))
coef_output <- data.frame(
  term=rownames(fixef(m)), fixef(m), row.names=NULL,
  n_observations=nrow(df),
  n_sentence_ids=counts[["n_sentence_ids"]],
  n_inference_clusters=counts[["n_inference_clusters"]],
  exclude_contrastive=exclude_contrastive
)
write.csv(
  coef_output,
  file.path(OUT, variant_filename("rq1_coef_maximal_results.csv",
                                  exclude_contrastive)),
  row.names=FALSE
)
