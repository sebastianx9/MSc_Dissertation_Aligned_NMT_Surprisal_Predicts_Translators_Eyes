#!/usr/bin/env Rscript

# Install analysis dependencies into R_LIBS_USER. Run through the accompanying
# Slurm job rather than compiling packages on a CSF login node.

packages <- c(
  "brms", "dplyr", "loo", "posterior", "lme4", "lmerTest",
  "ggplot2", "tidyr", "patchwork", "GGally", "bayesplot", "gridExtra"
)
missing <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (!length(missing)) {
  cat("All requested packages are already installed.\n")
  quit(status = 0)
}

cat("Installing: ", paste(missing, collapse = ", "), "\n", sep = "")
install.packages(
  missing, repos = "https://cloud.r-project.org",
  Ncpus = max(1L, as.integer(Sys.getenv("SLURM_NTASKS", "1")))
)

still_missing <- packages[
  !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(still_missing)) {
  stop("Installation incomplete: ", paste(still_missing, collapse = ", "))
}
cat("Package installation completed.\n")
