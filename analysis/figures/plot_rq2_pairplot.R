#!/usr/bin/env Rscript

# Descriptive pairplot for RQ2. This figure shows the raw joint distributions
# and correlations; it is not a substitute for the covariate-adjusted joint
# model or its condition-by-surprisal coefficients.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(GGally)
})

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]), mustWork = TRUE)
} else {
  normalizePath("analysis/figures/plot_rq2_pairplot.R", mustWork = TRUE)
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
    predictors %>% select(sentence_id, word_index, surprisal_soft,
                          mono_surprisal, log10_freq),
    by = c("sentence_id", "word_index")
  ) %>%
  filter(complete.cases(surprisal_soft, mono_surprisal, log10_freq)) %>%
  anti_join(tibble(sentence_id = "S003", word_index = 3L),
            by = c("sentence_id", "word_index")) %>%
  mutate(
    condition = factor(
      if_else(stage == "read", "Reading", "Translation"),
      levels = c("Reading", "Translation")
    ),
    c_nmt = as.numeric(scale(surprisal_soft)),
    c_mono = as.numeric(scale(mono_surprisal)),
    log_tfd = log(total_fixation_duration_ms)
  )

stopifnot(nrow(data) == 19732L)
plot_data <- data %>%
  transmute(
    condition,
    `c[nmt]` = c_nmt,
    `c[mono]` = c_mono,
    `log(TFD)` = log_tfd
  )

condition_colours <- c(Reading = "#0072B2", Translation = "#D55E00")
density_panel <- function(data, mapping, ...) {
  ggplot(data, mapping) +
    geom_density(aes(colour = condition, fill = condition),
                 alpha = .12, linewidth = .75) +
    theme_minimal(base_size = 11) +
    theme(panel.grid = element_blank())
}
point_panel <- function(data, mapping, ...) {
  ggplot(data, mapping) +
    geom_point(aes(alpha = condition), size = .34) +
    scale_alpha_manual(
      values = c(Reading = .045, Translation = .025),
      guide = "none"
    ) +
    theme_minimal(base_size = 11) +
    theme(panel.grid = element_blank())
}

p <- ggpairs(
  plot_data, columns = 2:4, aes(colour = condition),
  upper = list(continuous = wrap("cor", size = 3.4, stars = FALSE)),
  lower = list(continuous = point_panel),
  diag = list(continuous = density_panel),
  labeller = label_parsed,
  legend = c(2, 1)
) +
  scale_colour_manual(values = condition_colours, name = NULL) +
  scale_fill_manual(values = condition_colours, name = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(size = 10),
    legend.position = "bottom"
  )

output_path <- file.path(output_dir, "rq2_eda_pairplot_latest.pdf")
ggsave(output_path, p, width = 7.4, height = 6.8, device = "pdf")
cat(sprintf(
  "Saved %s | N=%d (read=%d, translate=%d)\n",
  output_path, nrow(data), sum(data$condition == "Reading"),
  sum(data$condition == "Translation")
))
