#!/usr/bin/env Rscript

# RQ1 translation-stage small multiples from the authoritative coefficient fit.
# Points are a fixed sample of component-plus-residual observations from an
# auxiliary random-intercept model; lines and ribbons are posterior fixed-effect
# contributions from the full Bayesian model.

suppressPackageStartupMessages({
  library(brms)
  library(posterior)
  library(dplyr)
  library(ggplot2)
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
fit_path <- if (length(args) >= 1) args[[1]] else
  Sys.getenv(
    "RQ1_COEF_FIT",
    unset = file.path(
      Sys.getenv("DISSERTATION_DATA_DIR", unset = "."),
      "brm_cache", "rq1_coef_maximal_v2.rds"
    )
  )
out_path <- if (length(args) >= 2) args[[2]] else
  file.path(
    Sys.getenv("DISSERTATION_FIGURE_DIR", unset = "."),
    "rq1_partial_smallmultiples.pdf"
  )

if (!file.exists(fit_path)) {
  stop("Latest RQ1 coefficient fit not found: ", fit_path)
}
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

fit <- readRDS(fit_path)
draws <- as_draws_df(fit)
model_data <- fit$data %>%
  mutate(
    ambiguity = factor(ambiguity),
    participant = factor(participant),
    sentence_id = factor(sentence_id)
  )

predictors <- tibble::tribble(
  ~term,     ~label,
  "c_nmt",  "Aligned NMT surprisal",
  "c_wlen", "Word length",
  "c_wpos", "Sentence position",
  "c_freq", "Word frequency"
)

x_grid <- seq(-2, 2, length.out = 101)
curves <- bind_rows(lapply(seq_len(nrow(predictors)), function(i) {
  term <- predictors$term[[i]]
  slope <- draws[[paste0("b_", term)]]
  bind_rows(lapply(x_grid, function(x) {
    contribution <- slope * x
    tibble(
      predictor = predictors$label[[i]],
      z = x,
      estimate = median(contribution),
      lower = unname(quantile(contribution, 0.025)),
      upper = unname(quantile(contribution, 0.975))
    )
  }))
})) %>%
  mutate(predictor = factor(predictor, levels = predictors$label))

visual_model <- lmer(
  log_tfd ~ c_wlen + c_wpos + c_freq + ambiguity + c_nmt +
    (1 | participant) + (1 | sentence_id),
  data = model_data,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

set.seed(42)
sample_index <- sample(
  seq_len(nrow(model_data)), min(nrow(model_data), 5000L), replace = FALSE
)
auxiliary_coefficients <- fixef(visual_model)
auxiliary_residuals <- residuals(visual_model)

partial_points <- bind_rows(lapply(seq_len(nrow(predictors)), function(i) {
  term <- predictors$term[[i]]
  tibble(
    predictor = predictors$label[[i]],
    z = model_data[[term]],
    partial = auxiliary_residuals +
      auxiliary_coefficients[[term]] * model_data[[term]],
    row_id = seq_len(nrow(model_data))
  ) %>%
    filter(row_id %in% sample_index)
})) %>%
  mutate(predictor = factor(predictor, levels = predictors$label))

blue <- "#0072B2"
p <- ggplot(curves, aes(x = z, y = estimate)) +
  geom_hline(yintercept = 0, colour = "grey72", linewidth = 0.35) +
  geom_vline(xintercept = 0, colour = "grey82", linewidth = 0.3) +
  geom_point(
    data = partial_points,
    aes(x = z, y = partial),
    inherit.aes = FALSE, colour = blue, size = 0.34, alpha = 0.065
  ) +
  geom_ribbon(
    aes(ymin = lower, ymax = upper),
    fill = blue, alpha = 0.13, colour = NA
  ) +
  geom_line(colour = blue, linewidth = 0.9) +
  facet_wrap(~ predictor, ncol = 2) +
  scale_x_continuous(breaks = -2:2) +
  coord_cartesian(xlim = c(-2, 2), ylim = c(-1.15, 1.15)) +
  labs(
    x = "Predictor value (SD from the translation-stage mean)",
    y = expression(Delta * " log(TFD) relative to predictor mean")
  ) +
  theme_minimal(base_size = 10.5) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = "grey91", linewidth = 0.3),
    strip.text = element_text(face = "bold", hjust = 0),
    strip.background = element_rect(fill = "grey96", colour = NA),
    axis.title = element_text(size = 10),
    plot.margin = margin(5.5, 7.5, 5.5, 5.5)
  )

ggsave(out_path, p, width = 7.4, height = 5.2, device = "pdf")
png_path <- sub("\\.pdf$", ".png", out_path, ignore.case = TRUE)
ggsave(png_path, p, width = 7.4, height = 5.2, dpi = 220, bg = "white")

cat("Saved RQ1 translation-stage small multiples:\n", out_path, "\n", png_path, "\n")
