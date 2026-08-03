#!/usr/bin/env Rscript

# Posterior-predictive and residual diagnostics for the primary coefficient
# models. This reads cached brms fits and performs no refitting.

suppressPackageStartupMessages({
  library(brms)
  library(bayesplot)
  library(ggplot2)
  library(gridExtra)
})

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
  "--output-dir",
  file.path(Sys.getenv("DISSERTATION_OUTPUT_DIR", data_dir), "assumptions")
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
cache_dir <- file.path(data_dir, "brm_cache")

specs <- data.frame(
  model = c("RQ1 TFD", "RQ2 joint TFD", "RQ3 FFD", "RQ3 GD",
            "RQ3 Go-past", "RQ3 RRT"),
  fit = c(
    "rq1_coef_maximal_v2.rds",
    "rq2_joint_maximal_v4.rds",
    "rq3coef_v4_long_total_FFD.rds",
    "rq3coef_v4_long_total_GD.rds",
    "rq3coef_v3_total_Go-past.rds",
    "rq3coef_v3_total_RRT.rds"
  ),
  stringsAsFactors = FALSE
)

missing <- file.path(cache_dir, specs$fit)
missing <- missing[!file.exists(missing)]
if (length(missing)) {
  stop("Missing cached coefficient fits: ", paste(missing, collapse = ", "))
}

skewness <- function(x) {
  x <- x[is.finite(x)]
  mean((x - mean(x))^3) / stats::sd(x)^3
}

statistic_vector <- function(x) {
  c(
    mean = mean(x), sd = sd(x), median = median(x),
    q05 = unname(quantile(x, .05)),
    q95 = unname(quantile(x, .95)), skewness = skewness(x)
  )
}

ppc_rows <- list()
residual_rows <- list()
residual_stage_rows <- list()

lag1_correlation <- function(residual, model_data) {
  required <- c("participant", "sentence_id")
  if (!all(required %in% names(model_data))) return(NA_real_)
  stage_name <- intersect(c("condition", "stage"), names(model_data))
  group_values <- list(
    participant = model_data$participant,
    sentence_id = model_data$sentence_id
  )
  if (length(stage_name)) group_values$stage <- model_data[[stage_name[[1]]]]
  group_id <- do.call(paste, c(group_values, sep = "\r"))
  pairs <- lapply(split(seq_along(residual), group_id), function(index) {
    # brms stores rows in the order supplied to the model but retains only
    # formula variables.  The analysis data were supplied in increasing word
    # order within participant-sentence-stage trials.  Prefer an explicit
    # word_index when available; otherwise preserve that stored within-trial
    # row order.
    if ("word_index" %in% names(model_data)) {
      index <- index[order(model_data$word_index[index])]
    } else {
      index <- sort(index)
    }
    if (length(index) < 2L) return(NULL)
    cbind(previous = residual[index[-length(index)]],
          current = residual[index[-1L]])
  })
  pairs <- pairs[lengths(pairs) > 0L]
  if (!length(pairs)) return(NA_real_)
  pair_matrix <- do.call(rbind, pairs)
  cor(pair_matrix[, "previous"], pair_matrix[, "current"],
      use = "complete.obs")
}

for (i in seq_len(nrow(specs))) {
  model_name <- specs$model[[i]]
  fit_name <- specs$fit[[i]]
  fit_path <- file.path(cache_dir, fit_name)
  cat("Loading ", fit_name, "\n", sep = "")
  fit <- readRDS(fit_path)
  if (!inherits(fit, "brmsfit")) stop("Not a brmsfit: ", fit_path)

  y <- as.numeric(brms::get_y(fit))
  if (!length(y) || any(!is.finite(y))) {
    stop("Observed response is empty or non-finite: ", fit_name)
  }
  set.seed(42)
  yrep <- posterior_predict(fit, ndraws = 200)
  fitted_mean <- fitted(fit, summary = TRUE)[, "Estimate"]
  residual <- y - fitted_mean
  model_data <- fit$data
  focal_predictors <- intersect(c("c_nmt", "c_mono"), names(model_data))

  observed_stats <- statistic_vector(y)
  replicated_stats <- t(apply(yrep, 1L, statistic_vector))
  p_upper <- colMeans(sweep(replicated_stats, 2L, observed_stats, `>=`))
  ppc_rows[[i]] <- data.frame(
    model = model_name,
    fit = fit_name,
    statistic = names(observed_stats),
    observed = as.numeric(observed_stats),
    replicated_mean = colMeans(replicated_stats),
    replicated_q2.5 = apply(replicated_stats, 2L, quantile, .025),
    replicated_q97.5 = apply(replicated_stats, 2L, quantile, .975),
    posterior_predictive_p_upper = as.numeric(p_upper),
    n_observations = length(y),
    stringsAsFactors = FALSE
  )

  qq_theoretical <- qnorm(ppoints(length(residual)))
  qq_observed <- sort(as.numeric(scale(residual)))
  residual_rows[[i]] <- data.frame(
    model = model_name,
    fit = fit_name,
    n_observations = length(y),
    residual_mean = mean(residual),
    residual_sd = sd(residual),
    residual_skewness = skewness(residual),
    correlation_abs_residual_fitted = cor(abs(residual), fitted_mean),
    normal_qq_correlation = cor(qq_theoretical, qq_observed),
    lag1_residual_correlation = lag1_correlation(residual, model_data),
    lag1_ordering = if ("word_index" %in% names(model_data)) {
      "word_index"
    } else {
      "stored within-trial row order"
    },
    correlation_residual_c_nmt = if ("c_nmt" %in% focal_predictors) {
      cor(residual, model_data$c_nmt, use = "complete.obs")
    } else NA_real_,
    correlation_residual_c_mono = if ("c_mono" %in% focal_predictors) {
      cor(residual, model_data$c_mono, use = "complete.obs")
    } else NA_real_,
    stringsAsFactors = FALSE
  )

  stage_name <- intersect(c("condition", "stage"), names(model_data))
  if (length(stage_name)) {
    stage_value <- as.character(model_data[[stage_name[[1]]]])
  } else {
    stage_value <- rep("single-stage model", length(residual))
  }
  stage_groups <- split(seq_along(residual), stage_value)
  residual_stage_rows[[i]] <- do.call(rbind, lapply(
    names(stage_groups),
    function(stage) {
      index <- stage_groups[[stage]]
      data.frame(
        model = model_name,
        fit = fit_name,
        stage = stage,
        n_observations = length(index),
        residual_mean = mean(residual[index]),
        residual_sd = sd(residual[index]),
        residual_variance = var(residual[index]),
        stringsAsFactors = FALSE
      )
    }
  ))

  plot_data <- data.frame(fitted = fitted_mean, residual = residual)
  p_density <- bayesplot::ppc_dens_overlay(y, yrep[1:50, , drop = FALSE]) +
    ggtitle(paste(model_name, "posterior predictive density")) +
    theme_minimal(base_size = 10)
  p_residual <- ggplot(plot_data, aes(fitted, residual)) +
    geom_hline(yintercept = 0, colour = "grey55", linetype = "dashed") +
    geom_point(alpha = .08, size = .35, colour = "#0072B2") +
    geom_smooth(method = "loess", se = FALSE, colour = "#D55E00",
                linewidth = .7) +
    labs(title = "Residuals versus fitted", x = "Posterior mean fitted value",
         y = "Observed - fitted") +
    theme_minimal(base_size = 10)
  qq_data <- data.frame(theoretical = qq_theoretical, observed = qq_observed)
  p_qq <- ggplot(qq_data, aes(theoretical, observed)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey55",
                linetype = "dashed") +
    geom_point(alpha = .12, size = .4, colour = "#0072B2") +
    labs(title = "Normal Q-Q plot of standardised residuals",
         x = "Theoretical quantile", y = "Residual quantile") +
    theme_minimal(base_size = 10)

  pdf(file.path(
    output_dir,
    paste0(gsub("[^A-Za-z0-9]+", "_", tolower(model_name)),
           "_diagnostics.pdf")
  ), width = 10, height = 3.5)
  gridExtra::grid.arrange(p_density, p_residual, p_qq, ncol = 3)
  dev.off()

  if (length(focal_predictors)) {
    pdf(file.path(
      output_dir,
      paste0(gsub("[^A-Za-z0-9]+", "_", tolower(model_name)),
             "_focal_linearity.pdf")
    ), width = 5.2, height = 3.8)
    for (predictor in focal_predictors) {
      focal_data <- data.frame(
        predictor = model_data[[predictor]], residual = residual
      )
      print(
        ggplot(focal_data, aes(predictor, residual)) +
          geom_hline(yintercept = 0, colour = "grey55",
                     linetype = "dashed") +
          geom_point(alpha = .08, size = .35, colour = "#0072B2") +
          geom_smooth(method = "loess", se = TRUE, colour = "#D55E00",
                      fill = "#D55E00", alpha = .15, linewidth = .7) +
          labs(
            title = paste(model_name, "residuals by", predictor),
            x = predictor, y = "Observed - fitted"
          ) +
          theme_minimal(base_size = 10)
      )
    }
    dev.off()
  }
}

ppc_output <- do.call(rbind, ppc_rows)
residual_output <- do.call(rbind, residual_rows)
residual_stage_output <- do.call(rbind, residual_stage_rows)
write.csv(ppc_output,
          file.path(output_dir, "posterior_predictive_summary.csv"),
          row.names = FALSE)
write.csv(residual_output,
          file.path(output_dir, "residual_summary.csv"), row.names = FALSE)
write.csv(residual_stage_output,
          file.path(output_dir, "residual_by_stage.csv"), row.names = FALSE)

cat("\nPosterior-predictive summaries\n")
print(ppc_output, row.names = FALSE)
cat("\nResidual summaries\n")
print(residual_output, row.names = FALSE)
cat("\nResidual summaries by stage\n")
print(residual_stage_output, row.names = FALSE)
cat("\nSaved diagnostics to ", output_dir, "\n", sep = "")
