# Pre-modelling screening of the c_nmt distribution before the S003/stoplight
# exclusion.  The standardisation reference is the exact translate-stage
# complete-case sample used by the current analysis pipeline.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]), mustWork = TRUE)
} else {
  normalizePath(
    "analysis/characterisation/screen_cnmt_copy_failures.R",
    mustWork = TRUE
  )
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
copy_scan_arg <- get_arg(
  "--copy-scan",
  Sys.getenv(
    "DISSERTATION_COPY_SCAN",
    unset = file.path(repo_root, "config", "copy_failures_full_scan.csv")
  )
)
copy_scan <- normalizePath(copy_scan_arg, mustWork = TRUE)

required <- file.path(data_dir, c(
  "fixation_durations_word.csv", "nmt_surprisal_soft_word.csv",
  "monolingual_surprisal_word.csv", "subtlex_us.csv"
))
missing <- required[!file.exists(required)]
if (length(missing)) stop("Missing inputs: ", paste(missing, collapse = ", "))
if (!file.exists(copy_scan)) stop("Missing copy-scan file: ", copy_scan)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

fix <- read.csv(required[[1]], stringsAsFactors = FALSE)
nmt <- read.csv(required[[2]], stringsAsFactors = FALSE)
mono <- read.csv(required[[3]], stringsAsFactors = FALSE)
freq <- read.delim(required[[4]], stringsAsFactors = FALSE, quote = "") %>%
  transmute(word_lower = lexical_form(Word), log10_freq = Lg10WF)

pred <- nmt %>%
  mutate(word_lower = lexical_form(word)) %>%
  left_join(freq, by = "word_lower") %>%
  left_join(
    mono %>% select(sentence_id, word_index,
                    mono_surprisal = surprisal_sum),
    by = c("sentence_id", "word_index")
  )

df <- fix %>%
  filter(stage == "translate") %>%
  left_join(
    pred %>% select(sentence_id, word_index, surprisal_soft,
                    mono_surprisal, log10_freq),
    by = c("sentence_id", "word_index")
  ) %>%
  filter(complete.cases(surprisal_soft, mono_surprisal, log10_freq)) %>%
  mutate(c_nmt = as.numeric(scale(surprisal_soft)))

wp <- df %>%
  distinct(sentence_id, word_index, c_nmt) %>%
  arrange(sentence_id, word_index) %>%
  mutate(idx = row_number())

copy <- read.csv(copy_scan, stringsAsFactors = FALSE) %>%
  select(sentence_id, word_index) %>%
  distinct() %>%
  left_join(wp, by = c("sentence_id", "word_index")) %>%
  filter(!is.na(c_nmt)) %>%
  mutate(status = if_else(
    sentence_id == "S003" & word_index == 3L, "excluded", "retained"
  ))
sl <- filter(copy, status == "excluded")
if (nrow(sl) != 1L) stop("Expected one S003/stoplight copy-scan position.")
next_largest <- max(wp$c_nmt[
  !(wp$sentence_id == "S003" & wp$word_index == 3L)
])

cat(sprintf(
  "obs=%d positions=%d | stoplight=%+.2f SD | next largest=%+.2f SD\n",
  nrow(df), nrow(wp), sl$c_nmt, next_largest
))

p <- ggplot(wp, aes(idx, c_nmt)) +
  geom_point(colour = "#0072B2", alpha = 0.35, size = 0.9) +
  geom_point(data = copy, colour = "#0072B2", size = 1.8,
             shape = 1, stroke = 0.8) +
  annotate(
    "text", x = sl$idx + 60, y = sl$c_nmt, colour = "grey30",
    size = 3.6, fontface = "italic",
    label = sprintf("stoplight (%+.2f SD)", sl$c_nmt), hjust = 0
  ) +
  labs(
    x = "Source-word positions (corpus order, S001 to S200)",
    y = expression(italic(c)[nmt] ~ "(SD units, before exclusion)")
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    axis.line.y = element_line(colour = "black", linewidth = 0.3)
  )

out <- file.path(output_dir, "rq1_cnmt_screening.pdf")
ggsave(out, p, width = 6.8, height = 3.4, device = "pdf")
cat("saved ", out, "\n", sep = "")
