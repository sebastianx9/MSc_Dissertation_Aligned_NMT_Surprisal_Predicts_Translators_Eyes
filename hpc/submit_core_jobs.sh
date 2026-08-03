#!/bin/bash

# Submit after csf3_check.sbatch has completed successfully.
# Usage: hpc/submit_core_jobs.sh /path/to/data [/path/to/output]
#
# The primary jobs retain S031/S032. A matched set of core sensitivity jobs
# refits the models after excluding the shared contrastive pair. The latter use
# a separate output directory, and the R scripts use distinct cache names.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 DATA_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

data_dir="$(cd "$1" && pwd)"
output_dir="${2:-${data_dir}/results}"
mkdir -p "${output_dir}"
sensitivity_output_dir="${output_dir}/exclude_contrastive"
mkdir -p "${sensitivity_output_dir}"
jobscript_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${jobscript_dir}/.." && pwd)"
jobscript="${jobscript_dir}/csf3_analysis.sbatch"
diagnostic_jobscript="${jobscript_dir}/csf3_diagnose_fit.sbatch"
assumption_jobscript="${jobscript_dir}/csf3_assumption_checks.sbatch"
repo_commit="$(git -C "${repo_dir}" rev-parse HEAD)"
if [[ -n "$(git -C "${repo_dir}" status --porcelain)" ]]; then
  echo "Commit or discard repository changes before submitting CSF jobs." >&2
  git -C "${repo_dir}" status --short >&2
  exit 2
fi
if [[ ! -f "${data_dir}/MANIFEST.sha256" ]]; then
  echo "Missing ${data_dir}/MANIFEST.sha256" >&2
  exit 2
fi
actual_manifest_sha256="$(sha256sum "${data_dir}/MANIFEST.sha256" | awk '{print $1}')"
if [[ -n "${EXPECTED_MANIFEST_SHA256:-}" && \
      "${actual_manifest_sha256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
  echo "Wrong input snapshot: expected manifest ${EXPECTED_MANIFEST_SHA256}, got ${actual_manifest_sha256}" >&2
  exit 2
fi
manifest_sha256="${EXPECTED_MANIFEST_SHA256:-${actual_manifest_sha256}}"
(
  cd "${data_dir}"
  sha256sum -c MANIFEST.sha256
)
echo "Input manifest SHA-256: ${manifest_sha256}"

submit() {
  local analysis="$1"
  local dependency="${2:-}"
  local exclude_contrastive="${3:-false}"
  local job_name="${4:-${analysis}}"
  local analysis_output_dir="${5:-${output_dir}}"
  local job_export="ALL,DISSERTATION_REPO_DIR=${repo_dir},DISSERTATION_DATA_DIR=${data_dir},DISSERTATION_OUTPUT_DIR=${analysis_output_dir},ANALYSIS=${analysis},EXCLUDE_CONTRASTIVE=${exclude_contrastive},EXPECTED_GIT_COMMIT=${repo_commit},EXPECTED_MANIFEST_SHA256=${manifest_sha256}"
  local args=(--parsable --job-name="${job_name}" --export="${job_export}")
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency="afterok:${dependency}")
  fi
  local submitted
  submitted="$(sbatch "${args[@]}" "${jobscript}")"
  printf '%s\n' "${submitted%%;*}"
}

submit_diagnostic() {
  local job_name="$1"
  local dependency="$2"
  local fit_path="$3"
  local diagnostic_output_dir="${output_dir}/diagnostics"
  local job_export="ALL,DISSERTATION_REPO_DIR=${repo_dir},DISSERTATION_DATA_DIR=${data_dir},BRMS_FIT_PATH=${fit_path},BRMS_DIAGNOSTIC_OUTPUT_DIR=${diagnostic_output_dir},EXPECTED_GIT_COMMIT=${repo_commit},EXPECTED_MANIFEST_SHA256=${manifest_sha256}"
  local submitted
  submitted="$(sbatch --parsable --job-name="${job_name}" \
    --dependency="afterok:${dependency}" --export="${job_export}" \
    "${diagnostic_jobscript}")"
  printf '%s\n' "${submitted%%;*}"
}

rq1_coef_id="$(submit rq1_coef)"
rq1_cv_id="$(submit rq1_cv)"
rq1_joint_id="$(submit rq1_joint_predictive "${rq1_cv_id}")"
rq1_robustness_id="$(submit rq1_robustness "${rq1_cv_id}")"
rq1_within_between_id="$(submit rq1_within_between "${rq1_coef_id}")"
rq1_reading_cmono_id="$(submit rq1_reading_cmono)"
rq1_reading_cnmt_id="$(submit rq1_reading_cnmt "${rq1_reading_cmono_id}")"
rq1_reading_joint_id="$(submit rq1_reading_joint "${rq1_reading_cnmt_id}")"
rq2_joint_id="$(submit rq2_joint)"
rq2_interaction_id="$(submit rq2_interaction_cv)"
rq2_stage_sigma_id="$(submit rq2_stage_sigma "${rq2_joint_id}")"
rq2_stoplight_id="$(submit rq2_joint_stoplight "${rq2_joint_id}")"
rq3_cv_id="$(submit rq3_cv)"
rq3_long_id="$(submit rq3_coefficient_long_refit "${rq3_cv_id}")"
rq1_coef_diagnostic_id="$(submit_diagnostic \
  rq1_coef_diagnostics "${rq1_coef_id}" \
  "${data_dir}/brm_cache/rq1_coef_maximal_v2.rds")"
rq2_joint_diagnostic_id="$(submit_diagnostic \
  rq2_joint_diagnostics "${rq2_joint_id}" \
  "${data_dir}/brm_cache/rq2_joint_maximal_v4.rds")"
rq2_sigma_diagnostic_id="$(submit_diagnostic \
  rq2_sigma_diagnostics "${rq2_stage_sigma_id}" \
  "${data_dir}/brm_cache/rq2_joint_stage_sigma_v1.rds")"
rq3_ffd_diagnostic_id="$(submit_diagnostic \
  rq3_ffd_diagnostics "${rq3_long_id}" \
  "${data_dir}/brm_cache/rq3coef_v4_long_total_FFD.rds")"
rq3_gd_diagnostic_id="$(submit_diagnostic \
  rq3_gd_diagnostics "${rq3_long_id}" \
  "${data_dir}/brm_cache/rq3coef_v4_long_total_GD.rds")"
rq3_go_past_diagnostic_id="$(submit_diagnostic \
  rq3_go_past_diagnostics "${rq3_cv_id}" \
  "${data_dir}/brm_cache/rq3coef_v3_total_Go-past.rds")"
rq3_rrt_diagnostic_id="$(submit_diagnostic \
  rq3_rrt_diagnostics "${rq3_cv_id}" \
  "${data_dir}/brm_cache/rq3coef_v3_total_RRT.rds")"
assumption_export="ALL,DISSERTATION_REPO_DIR=${repo_dir},DISSERTATION_DATA_DIR=${data_dir},DISSERTATION_OUTPUT_DIR=${output_dir},EXPECTED_GIT_COMMIT=${repo_commit},EXPECTED_MANIFEST_SHA256=${manifest_sha256}"
assumption_submitted="$(sbatch --parsable --job-name=model_assumptions \
  --dependency="afterok:${rq1_coef_id}:${rq2_joint_id}:${rq3_long_id}" \
  --export="${assumption_export}" "${assumption_jobscript}")"
assumption_id="${assumption_submitted%%;*}"

# Leave-pair-out sensitivity analyses. These are matched refits, not results
# obtained by subtracting S031/S032 from the primary pointwise ELPD values.
rq1_coef_excl_id="$(submit rq1_coef "${rq1_coef_id}" true rq1_coef_excl "${sensitivity_output_dir}")"
rq1_cv_excl_id="$(submit rq1_cv "${rq1_cv_id}" true rq1_cv_excl "${sensitivity_output_dir}")"
rq1_joint_excl_id="$(submit rq1_joint_predictive "${rq1_cv_excl_id}" true rq1_joint_excl "${sensitivity_output_dir}")"
rq2_joint_excl_id="$(submit rq2_joint "${rq2_joint_id}" true rq2_joint_excl "${sensitivity_output_dir}")"
rq2_interaction_excl_id="$(submit rq2_interaction_cv "${rq2_interaction_id}" true rq2_interaction_excl "${sensitivity_output_dir}")"
rq3_cv_excl_id="$(submit rq3_cv "${rq3_cv_id}" true rq3_cv_excl "${sensitivity_output_dir}")"
rq3_long_excl_id="$(submit rq3_coefficient_long_refit "${rq3_cv_excl_id}" true rq3_long_excl "${sensitivity_output_dir}")"
rq3_ffd_excl_diagnostic_id="$(submit_diagnostic \
  rq3_ffd_excl_diagnostics "${rq3_cv_excl_id}" \
  "${data_dir}/brm_cache/rq3coef_v3_total_FFD_exclude_contrastive.rds")"
rq3_gd_excl_diagnostic_id="$(submit_diagnostic \
  rq3_gd_excl_diagnostics "${rq3_long_excl_id}" \
  "${data_dir}/brm_cache/rq3coef_v4_long_total_GD_exclude_contrastive.rds")"
rq3_go_past_excl_diagnostic_id="$(submit_diagnostic \
  rq3_go_past_excl_diagnostics "${rq3_long_excl_id}" \
  "${data_dir}/brm_cache/rq3coef_v4_long_total_Go-past_exclude_contrastive.rds")"
rq3_rrt_excl_diagnostic_id="$(submit_diagnostic \
  rq3_rrt_excl_diagnostics "${rq3_cv_excl_id}" \
  "${data_dir}/brm_cache/rq3coef_v3_total_RRT_exclude_contrastive.rds")"

printf '%-24s %s\n' \
  rq1_coef "${rq1_coef_id}" \
  rq1_cv "${rq1_cv_id}" \
  rq1_joint_predictive "${rq1_joint_id}" \
  rq1_robustness "${rq1_robustness_id}" \
  rq1_within_between "${rq1_within_between_id}" \
  rq1_reading_cmono "${rq1_reading_cmono_id}" \
  rq1_reading_cnmt "${rq1_reading_cnmt_id}" \
  rq1_reading_joint "${rq1_reading_joint_id}" \
  rq2_joint "${rq2_joint_id}" \
  rq2_joint_stoplight "${rq2_stoplight_id}" \
  rq2_interaction_cv "${rq2_interaction_id}" \
  rq2_stage_sigma "${rq2_stage_sigma_id}" \
  rq3_cv "${rq3_cv_id}" \
  rq3_coefficient_long_refit "${rq3_long_id}" \
  rq1_coef_diagnostics "${rq1_coef_diagnostic_id}" \
  rq2_joint_diagnostics "${rq2_joint_diagnostic_id}" \
  rq2_sigma_diagnostics "${rq2_sigma_diagnostic_id}" \
  rq3_ffd_diagnostics "${rq3_ffd_diagnostic_id}" \
  rq3_gd_diagnostics "${rq3_gd_diagnostic_id}" \
  rq3_go_past_diagnostics "${rq3_go_past_diagnostic_id}" \
  rq3_rrt_diagnostics "${rq3_rrt_diagnostic_id}" \
  model_assumptions "${assumption_id}" \
  rq1_coef_exclude_pair "${rq1_coef_excl_id}" \
  rq1_cv_exclude_pair "${rq1_cv_excl_id}" \
  rq1_joint_exclude_pair "${rq1_joint_excl_id}" \
  rq2_joint_exclude_pair "${rq2_joint_excl_id}" \
  rq2_interaction_exclude_pair "${rq2_interaction_excl_id}" \
  rq3_cv_exclude_pair "${rq3_cv_excl_id}" \
  rq3_long_exclude_pair "${rq3_long_excl_id}"
printf '%-24s %s\n' \
  rq3_ffd_excl_diagnostics "${rq3_ffd_excl_diagnostic_id}" \
  rq3_gd_excl_diagnostics "${rq3_gd_excl_diagnostic_id}" \
  rq3_go_past_excl_diag "${rq3_go_past_excl_diagnostic_id}" \
  rq3_rrt_excl_diagnostics "${rq3_rrt_excl_diagnostic_id}"
