#!/usr/bin/env Rscript

# Summarise convergence diagnostics from a cached brms fit without refitting it.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  hit <- grep(paste0("^", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", name, "="), "", hit[[1]])
}

fit_path <- get_arg("--fit-path")
if (is.null(fit_path) || !nzchar(fit_path)) {
  stop("Supply --fit-path=/absolute/path/to/model.rds")
}
fit_path <- normalizePath(fit_path, mustWork = TRUE)

output_dir <- get_arg("--output-dir", dirname(fit_path))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)

required <- c("brms", "posterior")
missing <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing)) {
  stop("Missing R packages: ", paste(missing, collapse = ", "))
}

cat("Reading: ", fit_path, "\n", sep = "")
fit <- readRDS(fit_path)
if (!inherits(fit, "brmsfit")) {
  stop("The cached object is not a brmsfit: ", fit_path)
}
input_hashes <- attr(fit, "analysis_input_hashes", exact = TRUE)
if (is.null(input_hashes) || !length(input_hashes)) {
  stop("The cached fit lacks analysis-input hashes: ", fit_path)
}
input_hash_path <- file.path(
  output_dir,
  paste0(tools::file_path_sans_ext(basename(fit_path)), "_input_hashes.csv")
)
write.csv(
  data.frame(
    input = names(input_hashes), md5 = unname(input_hashes),
    stringsAsFactors = FALSE
  ),
  input_hash_path, row.names = FALSE
)

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

sampler_parameters <- brms::nuts_params(fit)
divergences <- sum(
  sampler_parameters$Value[sampler_parameters$Parameter == "divergent__"]
)
treedepth <- sampler_parameters$Value[
  sampler_parameters$Parameter == "treedepth__"
]
energy <- sampler_parameters[
  sampler_parameters$Parameter == "energy__", , drop = FALSE
]
bfmi_by_chain <- tapply(
  energy$Value, energy$Chain,
  function(values) mean(diff(values)^2) / stats::var(values)
)
configured_treedepth <- tryCatch(
  fit$fit@stan_args[[1]]$control$max_treedepth,
  error = function(error) NA_real_
)
if (is.null(configured_treedepth) || !length(configured_treedepth)) {
  configured_treedepth <- NA_real_
}
treedepth_hits <- if (is.finite(configured_treedepth)) {
  sum(treedepth >= configured_treedepth)
} else {
  NA_integer_
}

stem <- tools::file_path_sans_ext(basename(fit_path))
csv_path <- file.path(output_dir, paste0(stem, "_sampling_diagnostics.csv"))
write.csv(diagnostics, csv_path, row.names = FALSE)

n_draws <- posterior::ndraws(draws)
finite_rhat <- diagnostics$rhat[is.finite(diagnostics$rhat)]
finite_bulk <- diagnostics$ess_bulk[is.finite(diagnostics$ess_bulk)]
finite_tail <- diagnostics$ess_tail[is.finite(diagnostics$ess_tail)]

cat(sprintf(
  paste0(
    "Posterior draws: %d\nParameters diagnosed: %d\n",
    "Maximum R-hat: %.4f | >1.01: %d | >1.05: %d\n",
    "Minimum bulk ESS: %.1f | below 400: %d\n",
    "Minimum tail ESS: %.1f | below 400: %d\n"
  ),
  n_draws, nrow(diagnostics),
  max(finite_rhat), sum(finite_rhat > 1.01), sum(finite_rhat > 1.05),
  min(finite_bulk), sum(finite_bulk < 400),
  min(finite_tail), sum(finite_tail < 400)
))
cat(sprintf(
  paste0(
    "Divergences: %d\nObserved maximum treedepth: %.0f | ",
    "configured maximum: %s | hits: %s\nMinimum BFMI: %.4f\n"
  ),
  divergences,
  max(treedepth),
  ifelse(is.finite(configured_treedepth),
         as.character(configured_treedepth), "unavailable"),
  ifelse(is.na(treedepth_hits), "unavailable", as.character(treedepth_hits)),
  min(bfmi_by_chain, na.rm = TRUE)
))

summary_path <- file.path(
  output_dir, paste0(stem, "_sampler_summary.csv")
)
write.csv(
  data.frame(
    fit=basename(fit_path), posterior_draws=n_draws,
    max_rhat=max(finite_rhat), min_bulk_ess=min(finite_bulk),
    min_tail_ess=min(finite_tail), divergences=divergences,
    observed_max_treedepth=max(treedepth),
    configured_max_treedepth=configured_treedepth,
    treedepth_hits=treedepth_hits,
    min_bfmi=min(bfmi_by_chain, na.rm=TRUE),
    input_hash_count=length(input_hashes)
  ),
  summary_path, row.names=FALSE
)

print_rows <- function(title, data, order_by, decreasing = FALSE, n = 15L) {
  cat("\n", title, "\n", sep = "")
  keep <- is.finite(data[[order_by]])
  out <- data[keep, , drop = FALSE]
  out <- out[order(out[[order_by]], decreasing = decreasing), , drop = FALSE]
  print(utils::head(out, n), row.names = FALSE, digits = 4)
}

print_rows("Largest R-hat values:", diagnostics, "rhat", decreasing = TRUE)
print_rows("Lowest bulk ESS values:", diagnostics, "ess_bulk")
print_rows("Lowest tail ESS values:", diagnostics, "ess_tail")

key_parameters <- diagnostics[
  grepl("^(b_|Intercept$|sigma$|sd_|cor_)", diagnostics$variable),
  , drop = FALSE
]
print_rows(
  "Largest R-hat values among fixed effects and distribution/group parameters:",
  key_parameters, "rhat", decreasing = TRUE, n = 30L
)

cat("\nFull diagnostics written to: ", csv_path, "\n", sep = "")
cat("Sampler summary written to: ", summary_path, "\n", sep = "")
cat("Analysis input hashes written to: ", input_hash_path, "\n", sep = "")
cat("\nRandom-effects summary:\n")
print(brms::VarCorr(fit), digits=4)
