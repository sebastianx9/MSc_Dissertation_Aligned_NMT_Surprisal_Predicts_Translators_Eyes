#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=dissertation-r-packages
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
module purge
module load apps/gcc/R/4.4.1

repo_dir="${DISSERTATION_REPO_DIR:-${SLURM_SUBMIT_DIR}}"

export R_LIBS_USER="${R_LIBS_USER:-${HOME}/R/csf3-4.4}"
mkdir -p "${R_LIBS_USER}"
Rscript "${repo_dir}/hpc/install_r_packages.R"
