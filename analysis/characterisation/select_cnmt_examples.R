#!/usr/bin/env Rscript

# Descriptive word examples for the c_nmt / c_mono construct interpretation.
suppressPackageStartupMessages(library(dplyr))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: select_cnmt_examples.R POS_CSV OUTPUT_DIR [TOP_K]")
}
pos_path <- args[[1]]
output_dir <- args[[2]]
top_k <- if (length(args) >= 3) as.integer(args[[3]]) else 15L
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data <- read.csv(pos_path, stringsAsFactors = FALSE) |>
  mutate(
    word_clean = tolower(sub("[[:punct:]]+$", "", word)),
    quadrant = case_when(
      z_c_nmt > 0 & z_c_mono < 0 ~ "higher_nmt_lower_mono",
      z_c_nmt > 0.5 & z_c_mono > 0.5 ~ "both_high",
      z_c_nmt < 0 & z_c_mono < 0 ~ "both_low",
      TRUE ~ NA_character_
    )
  ) |>
  filter(!is.na(quadrant), nzchar(word_clean))

select_examples <- function(frame, name, descending = TRUE) {
  selected <- frame |>
    filter(quadrant == name) |>
    arrange(if (descending) desc(z_c_nmt) else z_c_nmt) |>
    distinct(word_clean, .keep_all = TRUE) |>
    slice_head(n = top_k) |>
    transmute(
      set = name,
      sentence_id,
      word_index,
      word = word_clean,
      pos,
      word_class,
      z_c_nmt,
      z_c_mono
    )
  selected
}

examples <- bind_rows(
  select_examples(data, "higher_nmt_lower_mono"),
  select_examples(data, "both_high"),
  select_examples(data, "both_low", descending = FALSE)
)
write.csv(
  examples,
  file.path(output_dir, "rq1_cnmt_word_examples.csv"),
  row.names = FALSE
)
print(examples, row.names = FALSE)
