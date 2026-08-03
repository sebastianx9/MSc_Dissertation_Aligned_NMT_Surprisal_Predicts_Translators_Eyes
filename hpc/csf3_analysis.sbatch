#!/bin/bash --login
#SBATCH -p multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=dissertation-model
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
module purge
module load apps/gcc/R/4.4.1

: "${ANALYSIS:?Set ANALYSIS before sbatch}"
: "${DISSERTATION_DATA_DIR:?Set DISSERTATION_DATA_DIR before sbatch}"
export R_LIBS_USER="${R_LIBS_USER:-${HOME}/R/csf3-4.4}"

exclude_contrastive="${EXCLUDE_CONTRASTIVE:-false}"
case "${exclude_contrastive}" in
  true|false) ;;
  *)
    echo "EXCLUDE_CONTRASTIVE must be true or false, got: ${exclude_contrastive}" >&2
    exit 2
    ;;
esac

repo_dir="${DISSERTATION_REPO_DIR:-${SLURM_SUBMIT_DIR}}"
output_dir="${DISSERTATION_OUTPUT_DIR:-${DISSERTATION_DATA_DIR}/results}"
mkdir -p "${output_dir}" "${DISSERTATION_DATA_DIR}/brm_cache"

actual_git_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
echo "Git commit: ${actual_git_commit}"
if [[ -n "$(git -C "${repo_dir}" status --porcelain)" ]]; then
  echo "Repository working tree is not clean; refusing a non-reproducible run." >&2
  git -C "${repo_dir}" status --short >&2
  exit 2
fi
if [[ -n "${EXPECTED_GIT_COMMIT:-}" && \
      "${actual_git_commit}" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "Repository changed after submission: expected ${EXPECTED_GIT_COMMIT}, got ${actual_git_commit}" >&2
  exit 2
fi
(
  cd "${DISSERTATION_DATA_DIR}"
  actual_manifest_sha256="$(sha256sum MANIFEST.sha256 | awk '{print $1}')"
  echo "Input manifest SHA-256: ${actual_manifest_sha256}"
  if [[ -n "${EXPECTED_MANIFEST_SHA256:-}" && \
        "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
    echo "Input manifest changed after submission: expected ${EXPECTED_MANIFEST_SHA256}, got ${actual_manifest_sha256}" >&2
    exit 2
  fi
  sha256sum -c MANIFEST.sha256
)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Analysis: ${ANALYSIS}"
echo "Repository: ${repo_dir}"
echo "Data: ${DISSERTATION_DATA_DIR}"
echo "Output: ${output_dir}"
echo "Exclude S031/S032: ${exclude_contrastive}"
Rscript -e 'print(sessionInfo())'
sha256sum \
  "${DISSERTATION_DATA_DIR}/fixation_durations_word.csv" \
  "${DISSERTATION_DATA_DIR}/eye_measures_word.csv" \
  "${DISSERTATION_DATA_DIR}/monolingual_surprisal_word.csv" \
  "${DISSERTATION_DATA_DIR}/nmt_surprisal_soft_word.csv" \
  "${DISSERTATION_DATA_DIR}/nmt_alignment_mass_word.csv" \
  "${DISSERTATION_DATA_DIR}/attention_features_6_norm.csv" \
  "${DISSERTATION_DATA_DIR}/subtlex_us.csv"

run_r() {
  local script="$1"
  shift
  Rscript "${script}" \
    "--data-dir=${DISSERTATION_DATA_DIR}" \
    "--output-dir=${output_dir}" \
    "--exclude-contrastive=${exclude_contrastive}" \
    "$@"
}

registry="${repo_dir}/config/analyses.tsv"
if [[ ! -f "${registry}" ]]; then
  echo "Missing analysis registry: ${registry}" >&2
  exit 2
fi
registry_line="$(awk -F $'\t' -v key="${ANALYSIS}" '
  NR > 1 && $1 == key { print; found = 1; exit }
  END { if (!found) exit 2 }
' "${registry}")" || {
  echo "Unknown ANALYSIS=${ANALYSIS}" >&2
  exit 2
}
IFS=$'\t' read -r registry_key script_relative fixed_args <<< "${registry_line}"
script_path="${repo_dir}/${script_relative}"
if [[ ! -f "${script_path}" ]]; then
  echo "Registered analysis script is missing: ${script_path}" >&2
  exit 2
fi
git -C "${repo_dir}" ls-files --error-unmatch "${script_relative}" >/dev/null

analysis_args=()
if [[ -n "${fixed_args}" ]]; then
  read -r -a analysis_args <<< "${fixed_args}"
fi
run_r "${script_path}" "${analysis_args[@]}"
