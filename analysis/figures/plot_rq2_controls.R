#!/usr/bin/env Rscript

# Posterior small multiples for the RQ2 joint model.
#
# Each panel isolates one continuous predictor's fixed-effect contribution to
# log(TFD), relative to that predictor's mean (z = 0).  Reading and translation
# slopes are calculated from the same posterior used for the coefficient table.
# This makes every condition-by-control term visible and avoids treating raw
# repeated observations as independent evidence.

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
  library(dplyr)
  library(ggplot2)
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
fit_path <- if (length(args) >= 1) args[[1]] else
  file.path(
    Sys.getenv("DISSERTATION_DATA_DIR", unset = "."),
    "brm_cache", "rq2_joint_maximal_v4.rds"
  )
out_path <- if (length(args) >= 2) args[[2]] else
  file.path(
    Sys.getenv("DISSERTATION_FIGURE_DIR", unset = "."),
    "rq2_control_smallmultiples_latest.pdf"
  )

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

fit <- readRDS(fit_path)
draws <- as_draws_df(fit)
model_data <- fit$data %>%
  mutate(
    condition = as.integer(condition),
    stage = factor(
      if_else(condition == 0L, "Reading", "Translation"),
      levels = c("Reading", "Translation")
    ),
    ambiguity = factor(ambiguity),
    participant = factor(participant),
    sentence_id = factor(sentence_id)
  )

predictors <- tibble::tribble(
  ~term,     ~label,
  "c_wlen", "Word length",
  "c_wpos", "Sentence position",
  "c_freq", "Word frequency"
)

x_grid <- seq(-2, 2, length.out = 101)

summarise_curve <- function(term, label, stage) {
  slope <- draws[[paste0("b_", term)]]
  if (stage == "Translation") {
    interaction_name <- paste0("b_condition:", term)
    if (!interaction_name %in% names(draws)) {
      stop("Missing interaction draw: ", interaction_name)
    }
    slope <- slope + draws[[interaction_name]]
  }

  bind_rows(lapply(x_grid, function(x) {
    contribution <- slope * x
    tibble(
      predictor = label,
      stage = stage,
      z = x,
      estimate = median(contribution),
      lower = unname(quantile(contribution, 0.025)),
      upper = unname(quantile(contribution, 0.975))
    )
  }))
}

curves <- bind_rows(lapply(seq_len(nrow(predictors)), function(i) {
  bind_rows(
    summarise_curve(predictors$term[[i]], predictors$label[[i]], "Reading"),
    summarise_curve(predictors$term[[i]], predictors$label[[i]], "Translation")
  )
})) %>%
  mutate(
    predictor = factor(predictor, levels = predictors$label),
    stage = factor(stage, levels = c("Reading", "Translation"))
  )

# The points are component-plus-residual observations from an auxiliary
# random-intercept fit with exactly the same fixed-effect structure.  They are
# descriptive; posterior inference and ribbons still come from the full brms
# model above.  A fixed stage-stratified sample keeps the vector figure legible.
visual_model <- lmer(
  log_tfd ~ condition *
    (c_wlen + c_wpos + c_freq + ambiguity + c_nmt + c_mono) +
    (1 | participant) + (1 | sentence_id),
  data = model_data,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

set.seed(42)
sample_index <- unlist(lapply(levels(model_data$stage), function(level) {
  candidates <- which(model_data$stage == level)
  sample(candidates, min(length(candidates), 3000L), replace = FALSE)
}))

auxiliary_coefficients <- fixef(visual_model)
auxiliary_residuals <- residuals(visual_model)

partial_points <- bind_rows(lapply(seq_len(nrow(predictors)), function(i) {
  term <- predictors$term[[i]]
  base_slope <- auxiliary_coefficients[[term]]
  interaction_slope <- auxiliary_coefficients[[paste0("condition:", term)]]
  stage_slope <- base_slope + model_data$condition * interaction_slope
  tibble(
    predictor = predictors$label[[i]],
    stage = model_data$stage,
    z = model_data[[term]],
    partial = auxiliary_residuals + stage_slope * model_data[[term]],
    row_id = seq_len(nrow(model_data))
  ) %>%
    filter(row_id %in% sample_index)
})) %>%
  mutate(
    predictor = factor(predictor, levels = predictors$label),
    stage = factor(stage, levels = c("Reading", "Translation"))
  )

stage_colours <- c("Reading" = "#0072B2", "Translation" = "#D55E00")

p <- ggplot(curves, aes(x = z, y = estimate, colour = stage, fill = stage)) +
  geom_hline(yintercept = 0, colour = "grey72", linewidth = 0.35) +
  geom_vline(xintercept = 0, colour = "grey82", linewidth = 0.3) +
  geom_point(
    data = partial_points,
    aes(x = z, y = partial, colour = stage),
    inherit.aes = FALSE, size = 0.32, alpha = 0.055,
    show.legend = FALSE
  ) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.11,
              colour = NA, show.legend = FALSE) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ predictor, ncol = 3) +
  scale_colour_manual(values = stage_colours, name = NULL) +
  scale_fill_manual(values = stage_colours, name = NULL) +
  scale_x_continuous(breaks = -2:2) +
  coord_cartesian(xlim = c(-2, 2), ylim = c(-1.15, 1.15)) +
  labs(
    x = "Predictor value (SD from the pooled mean)",
    y = expression(Delta * " log(TFD) relative to predictor mean")
  ) +
  theme_minimal(base_size = 10.5) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = "grey91", linewidth = 0.3),
    strip.text = element_text(face = "bold", hjust = 0),
    strip.background = element_rect(fill = "grey96", colour = NA),
    axis.title = element_text(size = 10),
    plot.margin = margin(5.5, 7.5, 5.5, 5.5)
  )

ggsave(out_path, p, width = 8.2, height = 3.15, device = "pdf")
png_path <- sub("\\.pdf$", ".png", out_path, ignore.case = TRUE)
ggsave(png_path, p, width = 8.2, height = 3.15, dpi = 220, bg = "white")

cat("Saved posterior slope small multiples:\n", out_path, "\n", png_path, "\n")
