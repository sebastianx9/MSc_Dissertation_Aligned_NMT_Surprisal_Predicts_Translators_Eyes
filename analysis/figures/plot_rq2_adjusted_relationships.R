#!/usr/bin/env Rscript

# Component-plus-residual visualisation for the RQ2 joint fixed-effect
# structure. Inference remains based on the Bayesian maximal model; the
# auxiliary random-intercept fit is used only to make the adjusted stage
# relationships visible at observation level.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(lme4)
})

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]), mustWork = TRUE)
} else {
  normalizePath("analysis/figures/plot_rq2_adjusted_relationships.R", mustWork = TRUE)
}
repo_root <- Sys.getenv("DISSERTATION_REPO_DIR", unset = "")
if (!nzchar(repo_root)) {
  repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
}
source(file.path(repo_root, "analysis", "shared", "analysis_design.R"))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  hit <- grep(paste0("^", name, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", name, "="), "", hit[[1]])
}
data_dir <- normalizePath(
  get_arg("--data-dir", Sys.getenv("DISSERTATION_DATA_DIR", unset = ".")),
  mustWork = TRUE
)
output_dir <- get_arg(
  "--output-dir", Sys.getenv("DISSERTATION_FIGURE_DIR", unset = ".")
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

fix <- read.csv(file.path(data_dir, "fixation_durations_word.csv"),
                stringsAsFactors = FALSE)
nmt <- read.csv(file.path(data_dir, "nmt_surprisal_soft_word.csv"),
                stringsAsFactors = FALSE)
mono <- read.csv(file.path(data_dir, "monolingual_surprisal_word.csv"),
                 stringsAsFactors = FALSE)
freq <- read.delim(file.path(data_dir, "subtlex_us.csv"),
                   stringsAsFactors = FALSE, quote = "") %>%
  transmute(word_lower = lexical_form(Word), log10_freq = Lg10WF)

sentence_lengths <- nmt %>%
  group_by(sentence_id) %>%
  summarise(sentence_length = max(word_index) + 1L, .groups = "drop")
predictors <- nmt %>%
  left_join(sentence_lengths, by = "sentence_id") %>%
  mutate(
    word_lower = lexical_form(word),
    word_length = nchar(word_lower, type = "chars"),
    word_position = word_index / (sentence_length - 1)
  ) %>%
  left_join(freq, by = "word_lower") %>%
  left_join(
    mono %>% select(sentence_id, word_index,
                    mono_surprisal = surprisal_sum),
    by = c("sentence_id", "word_index")
  )

data <- fix %>%
  filter(stage %in% c("read", "translate")) %>%
  left_join(
    predictors %>% select(sentence_id, word_index, word_length,
                          word_position, log10_freq,
                          nmt_surprisal = surprisal_soft, mono_surprisal),
    by = c("sentence_id", "word_index")
  ) %>%
  filter(complete.cases(nmt_surprisal, mono_surprisal, log10_freq)) %>%
  anti_join(tibble(sentence_id = "S003", word_index = 3L),
            by = c("sentence_id", "word_index")) %>%
  mutate(
    condition = factor(
      if_else(stage == "read", "Reading", "Translation"),
      levels = c("Reading", "Translation")
    ),
    ambiguity = factor(ambiguity),
    log_tfd = log(total_fixation_duration_ms),
    c_nmt = as.numeric(scale(nmt_surprisal)),
    c_mono = as.numeric(scale(mono_surprisal)),
    c_wlen = as.numeric(scale(word_length)),
    c_wpos = as.numeric(scale(word_position)),
    c_freq = as.numeric(scale(log10_freq))
  )
stopifnot(nrow(data) == 19732L)

visual_model <- lmer(
  log_tfd ~ condition *
    (c_wlen + c_wpos + c_freq + ambiguity + c_nmt + c_mono) +
    (1 | participant) + (1 | sentence_id),
  data = data, REML = FALSE,
  control = lmerControl(optimizer = "bobyqa")
)

component_residual <- function(model, data, predictor, label) {
  design <- model.matrix(model)
  coefficients <- fixef(model)
  columns <- grep(predictor, colnames(design), fixed = TRUE, value = TRUE)
  component <- as.numeric(
    design[, columns, drop = FALSE] %*% coefficients[columns]
  )
  tibble(
    condition = data$condition,
    predictor_value = data[[predictor]],
    partial_residual = residuals(model) + component +
      coefficients[["(Intercept)"]],
    predictor = label
  )
}

plot_data <- bind_rows(
  component_residual(
    visual_model, data, "c_nmt", "Aligned NMT surprisal"
  ),
  component_residual(
    visual_model, data, "c_mono", "Monolingual surprisal"
  )
) %>%
  mutate(
    predictor = factor(
      predictor,
      levels = c("Aligned NMT surprisal", "Monolingual surprisal")
    )
  )

condition_colours <- c(Reading = "#0072B2", Translation = "#D55E00")
p <- ggplot(
  plot_data,
  aes(predictor_value, partial_residual, colour = condition)
) +
  geom_point(aes(alpha = condition), size = .38) +
  geom_smooth(
    aes(fill = condition), method = "lm", se = TRUE,
    linewidth = .9, alpha = .14, show.legend = FALSE
  ) +
  facet_wrap(~predictor, scales = "free_x") +
  scale_colour_manual(values = condition_colours, name = NULL) +
  scale_fill_manual(values = condition_colours, guide = "none") +
  scale_alpha_manual(
    values = c(Reading = .045, Translation = .025), guide = "none"
  ) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 2))) +
  labs(
    x = "Predictor value (SD units)",
    y = "Component-plus-residual log(TFD)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black", linewidth = .3),
    strip.text = element_text(face = "plain"),
    legend.position = "top"
  )

output_path <- file.path(output_dir, "rq2_interaction_scatter_latest.pdf")
ggsave(output_path, p, width = 7, height = 4.25, device = "pdf")
cat(sprintf("Saved %s | N=%d\n", output_path, nrow(data)))
