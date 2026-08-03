#!/usr/bin/env Rscript

# RQ3 early-to-late phase profile.
#
# Panel A uses genuine posterior draws from the full-data coefficient models.
# Panel B uses within-outcome held-out ELPD gains and sentence-template-
# clustered intervals. FFD, GD, go-past, and conditional RRT are the RQ3
# profile; TFD is the aggregate RQ1 reference. No p-value or significance
# encoding is used.

suppressMessages({
  library(brms)
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  library(posterior)
})

args <- commandArgs(trailingOnly=TRUE)
get_arg <- function(name, default) {
  hit <- grep(paste0("^", name, "="), args, value=TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", name, "="), "", hit[[1]])
}

default_results_dir <- Sys.getenv(
  "DISSERTATION_RESULTS_DIR",
  Sys.getenv("DISSERTATION_OUTPUT_DIR", ".")
)
results_dir <- normalizePath(
  get_arg("--results-dir", get_arg("--data-dir", default_results_dir)),
  mustWork=TRUE
)
posterior_dir_arg <- get_arg(
  "--posterior-dir",
  Sys.getenv("DISSERTATION_POSTERIOR_DRAW_DIR", "")
)
posterior_dir <- if (nzchar(posterior_dir_arg)) {
  normalizePath(posterior_dir_arg, mustWork=TRUE)
} else {
  ""
}
default_data_root <- Sys.getenv(
  "DISSERTATION_DATA_DIR", dirname(results_dir)
)
fit_dir_arg <- get_arg(
  "--fit-dir", file.path(default_data_root, "brm_cache")
)
fit_dir <- if (nzchar(posterior_dir)) {
  fit_dir_arg
} else {
  normalizePath(fit_dir_arg, mustWork=TRUE)
}
output_dir <- get_arg(
  "--output-dir", Sys.getenv("DISSERTATION_FIGURE_DIR", results_dir)
)
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

rq3_path <- file.path(results_dir, "rq3_kfold_elpd.rds")
rq1_path <- file.path(results_dir, "rq1_kfold_elpd.rds")
missing_results <- c(rq3_path, rq1_path)[
  !file.exists(c(rq3_path, rq1_path))
]
if (length(missing_results)) {
  stop("Missing regenerated result files: ",
       paste(missing_results, collapse=", "))
}

rq3_object <- readRDS(rq3_path)
if (!is.list(rq3_object) || is.null(rq3_object$predictive)) {
  stop("RQ3 result predates the phase-localisation analysis.")
}
if (
  !is.null(rq3_object$inferential_scope) &&
    !identical(
      rq3_object$inferential_scope,
      "phase_profile_no_permutation"
    )
) {
  stop("RQ3 result has an incompatible inferential scope.")
}
rq3 <- rq3_object$predictive
rq1 <- readRDS(rq1_path)

rq3_required <- c(
  "outcome", "contrast", "elpd_diff", "se_cluster",
  "n_observations", "n_sentence_ids", "n_inference_clusters",
  "exclude_contrastive"
)
rq3_missing <- setdiff(rq3_required, names(rq3))
if (length(rq3_missing)) {
  stop("RQ3 result is missing columns: ",
       paste(rq3_missing, collapse=", "))
}
rq1_required <- c("res", "N", "J", "G", "exclude_contrastive")
rq1_missing <- setdiff(rq1_required, names(rq1))
if (length(rq1_missing)) {
  stop("RQ1 result is missing metadata: ",
       paste(rq1_missing, collapse=", "))
}
rq1_result_required <- c("predictor", "elpd_diff", "se_cluster")
rq1_result_missing <- setdiff(rq1_result_required, names(rq1$res))
if (length(rq1_result_missing)) {
  stop("RQ1 result table is missing columns: ",
       paste(rq1_result_missing, collapse=", "))
}

if (anyNA(rq3$exclude_contrastive) || any(rq3$exclude_contrastive)) {
  stop("The main RQ3 figure must not use the leave-pair-out result.")
}
if (!identical(rq1$exclude_contrastive, FALSE)) {
  stop("The main RQ3 figure must use the main RQ1 result.")
}

outcome_order <- c("FFD", "GD", "Go-past", "RRT", "TFD")
outcome_labels <- c(
  FFD="FFD",
  GD="GD",
  `Go-past`="Go-past",
  RRT="RRT (conditional)",
  TFD="TFD (aggregate)"
)
row_position <- setNames(rev(seq_along(outcome_order)), outcome_order)
authoritative_fit_names <- c(
  FFD="rq3coef_v4_long_total_FFD.rds",
  GD="rq3coef_v4_long_total_GD.rds",
  `Go-past`="rq3coef_v3_total_Go-past.rds",
  RRT="rq3coef_v3_total_RRT.rds",
  TFD="rq1_coef_maximal_v2.rds"
)
expected_draw_counts <- c(
  FFD=16000L,
  GD=16000L,
  `Go-past`=8000L,
  RRT=8000L,
  TFD=8000L
)

read_draw_file <- function(outcome) {
  if (nzchar(posterior_dir)) {
    path <- file.path(
      posterior_dir,
      sprintf("rq3_%s_c_nmt_draws.csv.gz", outcome)
    )
    if (outcome == "TFD") {
      path <- file.path(posterior_dir, "rq1_TFD_c_nmt_draws.csv.gz")
    }
    if (!file.exists(path)) {
      stop("Missing posterior draw file: ", path)
    }
    draw_data <- read.csv(path, stringsAsFactors=FALSE)
    if (!"c_nmt" %in% names(draw_data) || !nrow(draw_data)) {
      stop("Posterior draw file has no c_nmt draws: ", path)
    }
    if (outcome != "TFD") {
      if (!"fit_file" %in% names(draw_data)) {
        stop("RQ3 posterior draws lack fit provenance: ", path)
      }
      source_fits <- unique(draw_data$fit_file)
      if (!identical(source_fits, unname(authoritative_fit_names[[outcome]]))) {
        stop(
          "RQ3 posterior draws came from ", paste(source_fits, collapse=", "),
          "; expected ", authoritative_fit_names[[outcome]], ": ", path
        )
      }
    }
    if (nrow(draw_data) != expected_draw_counts[[outcome]]) {
      stop(
        "Unexpected posterior draw count for ", outcome, ": ",
        nrow(draw_data), " (expected ", expected_draw_counts[[outcome]], ")"
      )
    }
    values <- draw_data$c_nmt
  } else {
    fit_name <- unname(authoritative_fit_names[[outcome]])
    path <- file.path(fit_dir, fit_name)
    if (!file.exists(path)) {
      stop("Missing coefficient fit: ", path)
    }
    fit <- readRDS(path)
    draw_data <- posterior::as_draws_df(fit)
    if (!"b_c_nmt" %in% names(draw_data)) {
      stop("Coefficient fit has no b_c_nmt draws: ", path)
    }
    values <- draw_data[["b_c_nmt"]]
  }
  data.frame(
    outcome=outcome,
    c_nmt=values,
    stringsAsFactors=FALSE
  )
}

draws <- bind_rows(lapply(outcome_order, read_draw_file)) %>%
  mutate(
    y=unname(row_position[outcome]),
    series=ifelse(outcome == "TFD", "RQ1 aggregate", "RQ3 phase profile")
  )

draw_summary <- draws %>%
  group_by(outcome, y, series) %>%
  summarise(
    q025=quantile(c_nmt, .025),
    q25=quantile(c_nmt, .25),
    median=median(c_nmt),
    q75=quantile(c_nmt, .75),
    q975=quantile(c_nmt, .975),
    .groups="drop"
  )

make_density_polygon <- function(data, max_height=.31) {
  density_fit <- density(data$c_nmt, n=512, adjust=1.05)
  height <- density_fit$y / max(density_fit$y) * max_height
  base_y <- unique(data$y)
  stopifnot(length(base_y) == 1L)
  data.frame(
    outcome=unique(data$outcome),
    series=unique(data$series),
    x=c(density_fit$x[1], density_fit$x, density_fit$x[length(density_fit$x)]),
    y=c(base_y, base_y + height, base_y),
    stringsAsFactors=FALSE
  )
}
density_polygons <- bind_rows(
  lapply(split(draws, draws$outcome), make_density_polygon)
)

rq3_profile <- rq3 %>%
  filter(
    contrast == "c_nmt_total",
    outcome %in% c("FFD", "GD", "Go-past", "RRT")
  )
stopifnot(
  nrow(rq3_profile) == 4L,
  !anyDuplicated(rq3_profile$outcome),
  setequal(rq3_profile$outcome, c("FFD", "GD", "Go-past", "RRT")),
  all(rq3_profile$n_sentence_ids == 200L),
  all(rq3_profile$n_inference_clusters == 199L)
)
rq1_tfd <- rq1$res %>% filter(predictor == "c_nmt")
stopifnot(
  nrow(rq1_tfd) == 1L,
  rq1$N == 9047L,
  rq1$J == 200L,
  rq1$G == 199L,
  rq3_profile$n_observations[rq3_profile$outcome == "FFD"] == rq1$N,
  rq3_profile$n_observations[rq3_profile$outcome == "GD"] == rq1$N
)

predictive <- bind_rows(
  rq3_profile %>% transmute(
    outcome,
    elpd_diff,
    se_cluster,
    series="RQ3 phase profile"
  ),
  rq1_tfd %>% transmute(
    outcome="TFD",
    elpd_diff,
    se_cluster,
    series="RQ1 aggregate"
  )
) %>%
  mutate(
    y=unname(row_position[outcome]),
    lo=elpd_diff - 1.96 * se_cluster,
    hi=elpd_diff + 1.96 * se_cluster
  )

series_colours <- c(
  `RQ3 phase profile`="#0072B2",
  `RQ1 aggregate`="#6B7280"
)
common_theme <- theme_minimal(base_size=11.5) +
  theme(
    panel.grid.minor=element_blank(),
    panel.grid.major.y=element_blank(),
    plot.title=element_text(face="bold", size=11.5, hjust=0),
    axis.title.y=element_blank(),
    legend.position="none",
    plot.margin=margin(5.5, 7, 5.5, 5.5)
  )

panel_a <- ggplot() +
  geom_vline(xintercept=0, linetype="dashed", linewidth=.45,
             colour="grey55") +
  geom_polygon(
    data=density_polygons,
    aes(x=x, y=y, group=outcome, fill=series, colour=series),
    alpha=.28, linewidth=.35
  ) +
  geom_segment(
    data=draw_summary,
    aes(x=q025, xend=q975, y=y, yend=y, colour=series),
    linewidth=.65, lineend="round"
  ) +
  geom_segment(
    data=draw_summary,
    aes(x=q25, xend=q75, y=y, yend=y, colour=series),
    linewidth=2.1, lineend="round"
  ) +
  geom_point(
    data=draw_summary,
    aes(x=median, y=y, fill=series, colour=series, shape=series),
    size=2.2, stroke=.45
  ) +
  scale_colour_manual(values=series_colours) +
  scale_fill_manual(values=series_colours) +
  scale_shape_manual(values=c(`RQ3 phase profile`=21, `RQ1 aggregate`=23)) +
  scale_y_continuous(
    breaks=unname(row_position[outcome_order]),
    labels=unname(outcome_labels[outcome_order]),
    limits=c(.65, 5.48),
    expand=c(0, 0)
  ) +
  labs(
    title="A. Conditional association",
    x=expression("Standardised slope of " * c[nmt])
  ) +
  common_theme

panel_b <- ggplot(predictive, aes(x=elpd_diff, y=y, colour=series)) +
  geom_vline(xintercept=0, linetype="dashed", linewidth=.45,
             colour="grey55") +
  geom_segment(
    aes(x=lo, xend=hi, yend=y),
    linewidth=.8, lineend="round"
  ) +
  geom_point(
    aes(fill=series, shape=series),
    size=2.8, stroke=.55
  ) +
  scale_colour_manual(values=series_colours) +
  scale_fill_manual(values=series_colours) +
  scale_shape_manual(values=c(`RQ3 phase profile`=21, `RQ1 aggregate`=23)) +
  scale_y_continuous(
    breaks=unname(row_position[outcome_order]),
    labels=NULL,
    limits=c(.65, 5.48),
    expand=c(0, 0)
  ) +
  labs(
    title="B. Held-out predictive gain",
    x=expression(Delta * "ELPD (target - baseline)")
  ) +
  common_theme +
  theme(axis.text.y=element_blank(), axis.ticks.y=element_blank())

combined <- arrangeGrob(panel_a, panel_b, ncol=2, widths=c(1.08, 1))
figure_path <- file.path(output_dir, "rq3_outcome_profile_latest.pdf")
ggsave(
  figure_path, combined, width=9.1, height=4.65,
  device=pdf, units="in", useDingbats=FALSE
)
cat("Saved ", figure_path, "\n", sep="")
