import csv
import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


class AnalysisSubmissionTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO / relative_path).read_text(encoding="utf-8")

    def registry(self):
        with (REPO / "config" / "analyses.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def test_registry_is_unique_and_points_to_parseable_r_scripts(self):
        rows = self.registry()
        keys = [row["analysis"] for row in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreaterEqual(len(rows), 14)
        for row in rows:
            path = REPO / row["script"]
            self.assertTrue(path.is_file(), row)
            self.assertEqual(path.suffix, ".R")
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", row["script"]],
                cwd=REPO,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(tracked.returncode, 0, row)

    def test_rq2_interaction_cv_has_exactly_two_models(self):
        script = self.read("analysis/rq2/rq2_interaction_kfold.R")
        self.assertIn('common = f("c_mono + cond:c_mono + c_nmt")', script)
        self.assertIn("stage_specific =", script)
        self.assertIn("ptw$stage_specific - ptw$common", script)
        self.assertNotIn("M1 = f(", script)
        self.assertNotIn("N1 = f(", script)

    def test_rq1_joint_script_reports_all_three_comparisons(self):
        script = self.read("analysis/rq1/rq1_joint_surprisal_kfold.R")
        for contrast in (
            '"M_nmt - M_mono"',
            '"M_both - M_mono"',
            '"M_both - M_nmt"',
        ):
            self.assertIn(contrast, script)
        for cache_name in ("c_mono", "c_nmt", "c_mono_nmt"):
            self.assertIn(f'fit_kfold("{cache_name}"', script)

    def test_rq1_uses_all_six_lim_attention_features(self):
        script = self.read("analysis/rq1/rq1_kfold_elpd.R")
        extractor = self.read(
            "extraction/predictors/extract_attention_features.py"
        )
        self.assertIn("f_self=attn_self", script)
        self.assertIn('"c_fself"', script)
        self.assertIn('"attn_self"', extractor)

    def test_reading_profile_belongs_to_rq1_and_has_no_position_ablation(self):
        registry = {row["analysis"]: row for row in self.registry()}
        for key in (
            "rq1_reading_cmono",
            "rq1_reading_cnmt",
            "rq1_reading_joint",
        ):
            self.assertIn(key, registry)
            self.assertTrue(registry[key]["script"].startswith("analysis/rq1/"))
        cmono = self.read(
            "analysis/rq1/reading_profile/rq1_reading_cmono.R"
        )
        self.assertNotIn("base_without_position", cmono)
        self.assertNotIn("attenuation_when_position_is_added", cmono)

    def test_submission_graph_uses_registry_and_final_jobs(self):
        submit = self.read("hpc/submit_core_jobs.sh")
        runner = self.read("hpc/csf3_analysis.sbatch")
        self.assertIn("config/analyses.tsv", runner)
        registry_keys = {row["analysis"] for row in self.registry()}
        for key in registry_keys:
            self.assertIn(key, submit)
        self.assertNotIn("rq1_locus", submit)
        self.assertIn("EXPECTED_GIT_COMMIT", submit)
        self.assertIn("EXPECTED_MANIFEST_SHA256", submit)
        self.assertIn("rq3coef_v4_long_total_FFD.rds", submit)
        self.assertIn("rq3coef_v3_total_RRT.rds", submit)
        self.assertIn(
            "rq3coef_v4_long_total_GD_exclude_contrastive.rds", submit
        )
        self.assertIn(
            "rq3coef_v4_long_total_Go-past_exclude_contrastive.rds", submit
        )
        self.assertIn('rq3_long_excl_id="$(submit rq3_coefficient_long_refit', submit)

    def test_csf_preflight_verifies_the_input_manifest(self):
        check = self.read("hpc/csf3_check.sbatch")
        analysis = self.read("hpc/csf3_analysis.sbatch")
        assumptions = self.read("hpc/csf3_assumption_checks.sbatch")
        diagnostics = self.read("hpc/csf3_diagnose_fit.sbatch")
        for script in (check, analysis, assumptions, diagnostics):
            self.assertIn("sha256sum -c MANIFEST.sha256", script)
            self.assertIn("EXPECTED_MANIFEST_SHA256", script)
        self.assertIn("#SBATCH --ntasks=1", check)
        self.assertIn("#SBATCH --cpus-per-task=4", analysis)

    def test_diagnostics_record_analysis_input_hashes(self):
        diagnostics = self.read("hpc/diagnose_brms_fit.R")
        self.assertIn('attr(fit, "analysis_input_hashes"', diagnostics)
        self.assertIn('"_input_hashes.csv"', diagnostics)

    def test_rq3_is_a_phase_profile_without_permutation_inference(self):
        script = self.read("analysis/rq3/rq3_kfold_elpd.R")
        self.assertIn(
            'compare("c_nmt_total", "c_nmt", "base", "focal")',
            script,
        )
        self.assertNotIn('fitkf("c_mono"', script)
        self.assertNotIn("sign_flip", script)
        self.assertNotIn("p_holm", script)
        self.assertIn("phase_profile_no_permutation", script)
        self.assertIn("(1 + c_nmt | participant)", script)
        self.assertIn('Go_past=run_outcome(', script)

    def test_rq3_figure_uses_matching_total_tfd_reference(self):
        script = self.read("analysis/figures/plot_rq3_phase_profile.R")
        self.assertIn('"rq1_kfold_elpd.rds"', script)
        self.assertIn('predictor == "c_nmt"', script)
        self.assertIn('contrast == "c_nmt_total"', script)
        self.assertIn('"rq3_outcome_profile_latest.pdf"', script)

    def test_python_figure_scripts_use_a_headless_backend(self):
        paths = [
            "analysis/characterisation/plot_cnmt_construct.py",
            "extraction/figures/plot_alignment.py",
            "extraction/figures/plot_attention_features.py",
            "extraction/figures/plot_sample_trial.py",
        ]
        for path in paths:
            script = self.read(path)
            self.assertIn('use("Agg")', script, path)
            self.assertLess(script.index('use("Agg")'), script.index("pyplot"), path)

    def test_removed_historical_families_are_absent(self):
        for directory in ("RQ1", "RQ2", "RQ3", "data-extraction"):
            self.assertFalse((REPO / directory).exists())
        tracked_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for root in (REPO / "analysis", REPO / "hpc")
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".R", ".sh", ".sbatch"}
        ).lower()
        self.assertNotIn("rq_locus", tracked_text)
        self.assertNotIn("mono_neighbors", tracked_text)


if __name__ == "__main__":
    unittest.main()
