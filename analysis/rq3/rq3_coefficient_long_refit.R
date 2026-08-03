#!/usr/bin/env Rscript

# Targeted longer refits for RQ3 full-data coefficient models whose global
# sampler diagnostics exceeded R-hat 1.01.  The expensive grouped 10-fold CV
# fits are not rerun: this script only refits the stored coefficient models.

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
})

options(mc.cores = 4, warn = 1)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (!length(script_arg)) stop("Cannot determine the script path from --file.")
script_file <- normalizePath(
  sub("^--file=", "", script_arg[[1]]), mustWork = TRUE
)
repo_root <- Sys.getenv("DISSERTATION_REPO_DIR", unset = "")
if (!nzchar(repo_root)) {
  repo_root <- normalizePath(
    file.path(dirname(script_file), "..", ".."), mustWork = TRUE
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
  get_arg("--exclude-contrastive", "false"),
  "--exclude-contrastive"
)

# The primary FFD/GD fits and the leave-pair-out GD/Go-past fits were the only
# RQ3 coefficient objects with a global maximum R-hat above 1.01.  Their focal
# c_nmt slopes were already well mixed; these refits clean the whole-object
# diagnostics without changing the estimand.
default_outcomes <- if (exclude_contrastive) {
  c("GD", "Go-past")
} else {
  c("FFD", "GD")
}
outcomes_arg <- get_arg("--outcomes", paste(default_outcomes, collapse = ","))
outcomes <- trimws(strsplit(outcomes_arg, ",", fixed = TRUE)[[1]])
allowed_outcomes <- c("FFD", "GD", "Go-past", "RRT")
if (!length(outcomes) || any(!outcomes %in% allowed_outcomes)) {
  stop("--outcomes must contain only: ", paste(allowed_outcomes, collapse = ", "))
}

cache_dir <- file.path(data_dir, "brm_cache")
diagnostic_dir <- file.path(output_dir, "diagnostics")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(diagnostic_dir, recursive = TRUE, showWarnings = FALSE)

diagnose_fit <- function(fit, fit_name, source_md5) {
  draws <- posterior::as_draws_array(fit)
  diagnostics <- posterior::summarise_draws(
    draws, "rhat", "ess_bulk", "ess_tail"
  )
  diagnostics <- as.data.frame(diagnostics, stringsAsFactors = FALSE)
  diagnostics <- diagnostics[
    is.finite(diagnostics$rhat) |
      is.finite(diagnostics$ess_bulk) |
      is.finite(diagnostics$ess_tail),
    c("variable", "rhat", "ess_bulk", "ess_tail"),
    drop = FALSE
  ]

  sampler <- brms::nuts_params(fit)
  divergences <- sum(
    sampler$Value[sampler$Parameter == "divergent__"]
  )
  treedepth <- sampler$Value[sampler$Parameter == "treedepth__"]
  energy <- sampler[sampler$Parameter == "energy__", , drop = FALSE]
  bfmi <- tapply(
    energy$Value, energy$Chain,
    function(values) mean(diff(values)^2) / stats::var(values)
  )

  diagnostics_name <- paste0(
    tools::file_path_sans_ext(fit_name), "_sampling_diagnostics.csv"
  )
  write.csv(
    diagnostics, file.path(diagnostic_dir, diagnostics_name), row.names = FALSE
  )

  summary <- data.frame(
    fit = fit_name,
    source_fit_md5 = source_md5,
    posterior_draws = posterior::ndraws(draws),
    max_rhat = max(diagnostics$rhat, na.rm = TRUE),
    min_bulk_ess = min(diagnostics$ess_bulk, na.rm = TRUE),
    min_tail_ess = min(diagnostics$ess_tail, na.rm = TRUE),
    divergences = divergences,
    observed_max_treedepth = max(treedepth),
    configured_max_treedepth = 14L,
    treedepth_hits = sum(treedepth >= 14L),
    min_bfmi = min(bfmi, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  summary_name <- paste0(
    tools::file_path_sans_ext(fit_name), "_sampler_summary.csv"
  )
  write.csv(
    summary, file.path(diagnostic_dir, summary_name), row.names = FALSE
  )
  summary
}

refit_one <- function(outcome) {
  source_name <- variant_filename(
    sprintf("rq3coef_v3_total_%s.rds", outcome), exclude_contrastive
  )
  refit_name <- variant_filename(
    sprintf("rq3coef_v4_long_total_%s.rds", outcome), exclude_contrastive
  )
  source_path <- file.path(cache_dir, source_name)
  refit_path <- file.path(cache_dir, refit_name)
  if (!file.exists(source_path)) stop("Missing source fit: ", source_path)

  source_md5 <- unname(tools::md5sum(source_path))
  source_fit <- readRDS(source_path)
  if (!inherits(source_fit, "brmsfit")) {
    stop("Source object is not a brmsfit: ", source_path)
  }
  input_hashes <- attr(source_fit, "analysis_input_hashes", exact = TRUE)
  if (is.null(input_hashes)) {
    stop("Source fit lacks analysis-input hashes: ", source_path)
  }

  if (file.exists(refit_path)) {
    cat("Loading cached longer refit: ", refit_path, "\n", sep = "")
    fit <- readRDS(refit_path)
    assert_analysis_input_hashes(fit, input_hashes, refit_path)
    if (!identical(attr(fit, "source_fit_md5", exact = TRUE), source_md5)) {
      stop("Longer refit was produced from a different source fit: ", refit_path)
    }
  } else {
    cat("Longer refit for ", outcome, " from ", source_name, "\n", sep = "")
    fit <- update(
      source_fit,
      chains = 4, iter = 8000, warmup = 4000,
      seed = 42, refresh = 0, silent = 2,
      control = list(adapt_delta = 0.99, max_treedepth = 14)
    )
    fit <- set_analysis_input_hashes(fit, input_hashes)
    attr(fit, "source_fit_md5") <- source_md5
    attr(fit, "long_refit_specification") <- list(
      chains = 4L, iter = 8000L, warmup = 4000L, seed = 42L,
      adapt_delta = 0.99, max_treedepth = 14L
    )
    saveRDS(fit, refit_path)
  }

  fixed <- brms::fixef(fit)
  if (!"c_nmt" %in% rownames(fixed)) {
    stop("Longer refit has no c_nmt fixed effect: ", refit_path)
  }
  focal <- posterior::summarise_draws(
    posterior::as_draws_array(fit, variable = "b_c_nmt"),
    "rhat", "ess_bulk", "ess_tail"
  )
  sampler_summary <- diagnose_fit(fit, refit_name, source_md5)

  data.frame(
    outcome = outcome,
    term = "c_nmt",
    Estimate = fixed["c_nmt", "Estimate"],
    Est.Error = fixed["c_nmt", "Est.Error"],
    Q2.5 = fixed["c_nmt", "Q2.5"],
    Q97.5 = fixed["c_nmt", "Q97.5"],
    Rhat = focal$rhat,
    Bulk_ESS = focal$ess_bulk,
    Tail_ESS = focal$ess_tail,
    n_observations = nrow(fit$data),
    max_rhat = sampler_summary$max_rhat,
    min_bulk_ess = sampler_summary$min_bulk_ess,
    min_tail_ess = sampler_summary$min_tail_ess,
    divergences = sampler_summary$divergences,
    treedepth_hits = sampler_summary$treedepth_hits,
    min_bfmi = sampler_summary$min_bfmi,
    exclude_contrastive = exclude_contrastive,
    source_fit = source_name,
    source_fit_md5 = source_md5,
    refit = refit_name,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

results <- do.call(rbind, lapply(outcomes, refit_one))
output_name <- variant_filename(
  "rq3_coefficient_long_refit_results.csv", exclude_contrastive
)
write.csv(results, file.path(output_dir, output_name), row.names = FALSE)
cat("\nRQ3 targeted longer-refit results\n")
print(results, row.names = FALSE)
cat("Saved ", file.path(output_dir, output_name), "\n", sep = "")
