#!/usr/bin/env Rscript

# RQ1 held-out gains for the eight candidate predictors.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  hit <- grep(paste0("^", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", name, "="), "", hit[[1]])
}

results_dir <- normalizePath(
  get_arg(
    "--results-dir",
    Sys.getenv("DISSERTATION_RESULTS_DIR", unset =
      Sys.getenv("DISSERTATION_OUTPUT_DIR", unset = "."))
  ),
  mustWork = TRUE
)
output_dir <- get_arg(
  "--output-dir", Sys.getenv("DISSERTATION_FIGURE_DIR", unset = results_dir)
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

result_path <- file.path(results_dir, "rq1_kfold_elpd.rds")
result <- readRDS(result_path)
if (!is.list(result) || is.null(result$res)) {
  stop("Unexpected RQ1 result object: ", result_path)
}

plot_data <- result$res %>%
  mutate(
    group = if_else(
      predictor %in% c("c_nmt", "c_mono"), "Surprisal", "Attention"
    ),
    label = recode(
      predictor,
      c_nmt = "c[nmt]", c_mono = "c[mono]", H_e = "H[e]",
      f_e = "f[e]", f_self = "f[self]", f_eos = "f[eos]",
      f_recv = "f[recv]", f_cross = "f[cross]"
    ),
    label = factor(
      label,
      levels = c(
        "c[nmt]", "c[mono]", "H[e]", "f[e]", "f[self]",
        "f[eos]", "f[recv]", "f[cross]"
      )
    ),
    supported = p_holm < .05,
    lower = elpd_diff - 1.96 * se_cluster,
    upper = elpd_diff + 1.96 * se_cluster
  )

colours <- c(Surprisal = "#0072B2", Attention = "#D55E00")
plot <- ggplot(plot_data, aes(label, elpd_diff, colour = group)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey55") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0, linewidth = .75) +
  geom_point(
    aes(shape = supported), size = 3.2, fill = "white", stroke = 1.1
  ) +
  scale_shape_manual(
    values = c(`TRUE` = 16, `FALSE` = 21),
    labels = c(`TRUE` = "Holm p < .05", `FALSE` = "Holm p >= .05"),
    name = NULL
  ) +
  scale_colour_manual(values = colours, name = NULL) +
  scale_x_discrete(labels = function(value) parse(text = value)) +
  labs(
    x = NULL,
    y = expression(Delta * "ELPD over lexical/positional controls")
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position = "bottom",
    axis.line.y = element_line(colour = "black", linewidth = .3)
  )

output_path <- file.path(output_dir, "rq1_predictor_comparison_latest.pdf")
ggsave(output_path, plot, width = 7, height = 4.5, device = "pdf")
message("Saved: ", output_path)
