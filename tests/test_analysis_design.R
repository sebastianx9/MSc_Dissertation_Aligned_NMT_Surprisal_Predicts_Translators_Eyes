#!/usr/bin/env Rscript

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])
repo_root <- normalizePath(file.path(dirname(script_file), ".."), mustWork=TRUE)
source(file.path(repo_root, "analysis", "shared", "analysis_design.R"))

lexical_examples <- c(
  " Word. ", "DON'T", "state-of-the-art", "“Quoted”", "—dash—",
  "(nested?!", NA_character_
)
stopifnot(identical(
  lexical_form(lexical_examples),
  c("word", "don't", "state-of-the-art", "quoted", "dash", "nested",
    NA_character_)
))

sentence_id <- rep(sprintf("S%03d", seq_len(32)), each=3)
full_folds <- make_sentence_folds(sentence_id, K=10, seed=42)
pair_rows <- sentence_id %in% CONTRASTIVE_SENTENCE_IDS

stopifnot(
  length(unique(full_folds[pair_rows])) == 1L,
  length(unique(contrastive_group_id(sentence_id))) == 31L,
  variant_filename("result.rds", FALSE) == "result.rds",
  variant_filename("result.rds", TRUE) ==
    "result_exclude_contrastive.rds"
)

keep <- contrastive_keep(sentence_id, TRUE)
retained_folds <- full_folds[keep]
value <- seq_along(sentence_id)
scaled_full <- as.numeric(scale(value))
stopifnot(
  !any(sentence_id[keep] %in% CONTRASTIVE_SENTENCE_IDS),
  identical(retained_folds, full_folds[keep]),
  identical(scaled_full[keep], as.numeric(scale(value))[keep])
)

pointwise <- seq_along(sentence_id)
clustered <- cluster_delta_sums(pointwise, sentence_id)
stopifnot(
  length(clustered) == 31L,
  unname(clustered[CONTRASTIVE_CLUSTER_ID]) == sum(pointwise[pair_rows])
)

helper_path <- file.path(repo_root, "analysis", "shared", "analysis_design.R")
hashes <- analysis_input_hashes(c(design_helper=helper_path))
cached <- set_analysis_input_hashes(list(value=1L), hashes)
stopifnot(
  length(hashes) == 1L,
  identical(names(hashes), "design_helper"),
  identical(assert_analysis_input_hashes(cached, hashes), invisible(TRUE))
)
bad_hashes <- hashes
bad_hashes[] <- strrep("0", 32)
stopifnot(inherits(
  try(assert_analysis_input_hashes(cached, bad_hashes), silent=TRUE),
  "try-error"
))

cat("analysis-design tests passed\n")
