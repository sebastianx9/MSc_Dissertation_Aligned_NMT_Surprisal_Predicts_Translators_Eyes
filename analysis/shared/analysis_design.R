# Shared design helpers for the dissertation analyses.
#
# S031 and S032 are distinct sentence IDs and remain distinct in model random
# effects.  They are, however, a near-minimal contrastive pair shown to the
# same participants.  Treating them as one grouping unit prevents one member
# from leaking into training while the other is held out, and treats their
# paired sampling structure as one unit for clustered uncertainty.

CONTRASTIVE_SENTENCE_IDS <- c("S031", "S032")
CONTRASTIVE_CLUSTER_ID <- "__contrastive_pair_S031_S032__"

# Canonical form used for lexical controls.  Source tokens retain attached
# sentence punctuation (for example, "word."), whereas SUBTLEX lists the
# lexical item without it.  Remove Unicode punctuation only at token edges;
# apostrophes and hyphens inside a word remain part of the lexical form.
lexical_form <- function(word) {
  value <- trimws(as.character(word))
  value <- gsub("^\\p{P}+|\\p{P}+$", "", value, perl = TRUE)
  tolower(value)
}

parse_bool <- function(value, name = "boolean option") {
  value <- tolower(trimws(as.character(value)))
  if (value %in% c("true", "t", "1", "yes", "y")) return(TRUE)
  if (value %in% c("false", "f", "0", "no", "n")) return(FALSE)
  stop(name, " must be true or false; received: ", value)
}

analysis_variant_suffix <- function(exclude_contrastive) {
  if (isTRUE(exclude_contrastive)) "_exclude_contrastive" else ""
}

variant_filename <- function(filename, exclude_contrastive) {
  suffix <- analysis_variant_suffix(exclude_contrastive)
  if (!nzchar(suffix)) return(filename)
  extension <- tools::file_ext(filename)
  if (!nzchar(extension)) return(paste0(filename, suffix))
  stem <- substr(filename, 1L, nchar(filename) - nchar(extension) - 1L)
  paste0(stem, suffix, ".", extension)
}

contrastive_group_id <- function(sentence_id) {
  id <- as.character(sentence_id)
  id[id %in% CONTRASTIVE_SENTENCE_IDS] <- CONTRASTIVE_CLUSTER_ID
  id
}

apply_contrastive_sensitivity <- function(data, exclude_contrastive,
                                           sentence_column = "sentence_id") {
  if (!sentence_column %in% names(data)) {
    stop("Missing sentence column: ", sentence_column)
  }
  keep <- contrastive_keep(data[[sentence_column]], exclude_contrastive)
  data[keep, , drop = FALSE]
}

contrastive_keep <- function(sentence_id, exclude_contrastive) {
  if (!isTRUE(exclude_contrastive)) {
    return(rep(TRUE, length(sentence_id)))
  }
  !(as.character(sentence_id) %in% CONTRASTIVE_SENTENCE_IDS)
}

make_sentence_folds <- function(sentence_id, K = 10L, seed = 42L) {
  sentence_id <- as.character(sentence_id)
  group_id <- contrastive_group_id(sentence_id)
  set.seed(seed)
  folds <- loo::kfold_split_grouped(K = K, x = group_id)

  stopifnot(
    length(folds) == length(sentence_id),
    all(tapply(folds, sentence_id,
               function(x) length(unique(x))) == 1L),
    all(tapply(folds, group_id,
               function(x) length(unique(x))) == 1L)
  )
  folds
}

cluster_delta_sums <- function(pointwise_delta, sentence_id) {
  stopifnot(length(pointwise_delta) == length(sentence_id))
  tapply(pointwise_delta, contrastive_group_id(sentence_id), sum)
}

design_counts <- function(sentence_id) {
  sentence_id <- as.character(sentence_id)
  c(
    n_sentence_ids = length(unique(sentence_id)),
    n_inference_clusters = length(unique(contrastive_group_id(sentence_id)))
  )
}

assert_contrastive_fold_binding <- function(folds, sentence_id) {
  present <- as.character(sentence_id) %in% CONTRASTIVE_SENTENCE_IDS
  if (any(present)) {
    if (!all(CONTRASTIVE_SENTENCE_IDS %in% as.character(sentence_id))) {
      stop("Only one member of the S031/S032 contrastive pair is present.")
    }
    if (length(unique(folds[present])) != 1L) {
      stop("S031 and S032 must be assigned to the same CV fold.")
    }
  }
  invisible(TRUE)
}

analysis_input_hashes <- function(paths) {
  if (!length(paths)) stop("At least one input path is required.")
  path_names <- names(paths)
  paths <- as.character(paths)
  if (is.null(path_names) || any(!nzchar(path_names))) {
    path_names <- basename(paths)
  }
  if (anyDuplicated(path_names)) {
    stop("Input-hash labels must be unique: ",
         paste(path_names[duplicated(path_names)], collapse = ", "))
  }
  missing <- paths[!file.exists(paths)]
  if (length(missing)) {
    stop("Cannot hash missing input files: ", paste(missing, collapse = ", "))
  }
  hashes <- unname(tools::md5sum(paths))
  if (anyNA(hashes)) {
    stop("Failed to compute MD5 for: ",
         paste(paths[is.na(hashes)], collapse = ", "))
  }
  stats::setNames(as.character(hashes), path_names)
}

set_analysis_input_hashes <- function(object, input_hashes) {
  stopifnot(length(input_hashes) > 0L, !is.null(names(input_hashes)))
  attr(object, "analysis_input_hashes") <- input_hashes
  object
}

assert_analysis_input_hashes <- function(object, expected,
                                         label = "cached object") {
  actual <- attr(object, "analysis_input_hashes", exact = TRUE)
  if (is.null(actual)) {
    stop(label, " has no input-hash metadata; delete it and refit.")
  }
  if (!identical(actual, expected)) {
    all_names <- union(names(expected), names(actual))
    differences <- vapply(all_names, function(name) {
      expected_value <- if (name %in% names(expected)) {
        expected[[name]]
      } else {
        "<missing>"
      }
      actual_value <- if (name %in% names(actual)) {
        actual[[name]]
      } else {
        "<missing>"
      }
      if (identical(expected_value, actual_value)) return(NA_character_)
      paste0(name, " (cached=", actual_value,
             ", current=", expected_value, ")")
    }, character(1))
    differences <- differences[!is.na(differences)]
    stop(label, " was created from different inputs: ",
         paste(differences, collapse = "; "),
         ". Delete it and refit.")
  }
  invisible(TRUE)
}
