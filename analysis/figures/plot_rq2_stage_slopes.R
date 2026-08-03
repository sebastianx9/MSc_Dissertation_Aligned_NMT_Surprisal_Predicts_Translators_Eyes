#!/usr/bin/env Rscript

# RQ2 stage-specific posterior slopes for both surprisal measures.
suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(ggplot2)
  library(posterior)
  library(tidyr)
})

data_dir <- Sys.getenv("DISSERTATION_DATA_DIR", unset = getwd())
output_dir <- Sys.getenv(
  "DISSERTATION_FIGURE_DIR",
  unset = file.path(data_dir, "results", "figures")
)
fit_path <- Sys.getenv(
  "RQ2_JOINT_FIT",
  unset = file.path(data_dir, "brm_cache", "rq2_joint_maximal_v4.rds")
)

if (!file.exists(fit_path)) {
  stop("Latest RQ2 fit not found: ", fit_path)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

draws <- posterior::as_draws_df(readRDS(fit_path))
posterior_slopes <- tibble(
  `Aligned NMT surprisal` = draws$b_c_nmt,
  `Aligned NMT surprisal__Translation` =
    draws$b_c_nmt + draws[["b_condition:c_nmt"]],
  `Monolingual surprisal` = draws$b_c_mono,
  `Monolingual surprisal__Translation` =
    draws$b_c_mono + draws[["b_condition:c_mono"]]
) |>
  rename(
    `Aligned NMT surprisal__Reading` = `Aligned NMT surprisal`,
    `Monolingual surprisal__Reading` = `Monolingual surprisal`
  ) |>
  pivot_longer(everything(), names_to = "key", values_to = "slope") |>
  separate_wider_delim(key, delim = "__", names = c("Predictor", "Stage")) |>
  mutate(
    Predictor = factor(
      Predictor,
      levels = c("Aligned NMT surprisal", "Monolingual surprisal")
    ),
    Stage = factor(Stage, levels = c("Reading", "Translation"))
  )

plot <- ggplot(posterior_slopes, aes(Stage, slope, fill = Stage)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_violin(alpha = 0.48, colour = NA, width = 0.82, trim = FALSE) +
  stat_summary(
    fun = median,
    fun.min = function(x) quantile(x, 0.025),
    fun.max = function(x) quantile(x, 0.975),
    geom = "pointrange",
    linewidth = 0.55,
    size = 0.35,
    colour = "grey20"
  ) +
  facet_wrap(vars(Predictor), nrow = 1) +
  scale_fill_manual(values = c(Reading = "#0072B2", Translation = "#D55E00")) +
  labs(x = NULL, y = "Stage-specific slope (log-ms per SD)") +
  guides(fill = "none") +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    panel.spacing = grid::unit(1.1, "lines")
  )

figure_path <- file.path(output_dir, "rq2_posterior_slopes_latest.pdf")
ggsave(figure_path, plot, width = 6.6, height = 3.7, device = "pdf")
message("Saved: ", figure_path)
