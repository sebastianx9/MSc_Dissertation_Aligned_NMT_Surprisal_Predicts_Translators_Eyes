import sys
import tempfile
import unittest
from pathlib import Path
from statistics import median


EXTRACTION_DIR = (
    Path(__file__).resolve().parents[1] / "extraction" / "eye_tracking"
)
sys.path.insert(0, str(EXTRACTION_DIR))

from gaze_line import (
    _CandidateLine,
    _fixed_slope_refit,
    FixationBout,
    LineModel,
    estimate_text_line,
    fit_text_line,
    read_fixation_bouts,
    x_to_word_index,
)


def bout(sequence, x, y, duration=150.0):
    return FixationBout(
        sequence=sequence,
        start_s=float(sequence),
        end_s=float(sequence) + duration / 1000,
        mean_x=float(x),
        mean_y=float(y),
        duration_ms=float(duration),
    )


class TrialLineEstimationTests(unittest.TestCase):
    @staticmethod
    def ranges():
        return [
            (260 + i * 100, 340 + i * 100, str(i)) for i in range(6)
        ]

    def test_recovers_vertically_shifted_text_line(self):
        shifted_line = [
            bout(0, 300, 398),
            bout(1, 410, 405),
            bout(2, 530, 412),
            bout(3, 660, 407),
            bout(4, 790, 401),
            bout(5, 825, 409),
        ]
        retained, diagnostic = estimate_text_line(shifted_line, 260, 830)

        self.assertEqual(diagnostic.status, "ok")
        self.assertEqual(len(retained), 6)
        self.assertGreater(diagnostic.line_y, 390)
        self.assertLess(diagnostic.line_y, 420)

    def test_rejects_off_line_fixations_even_when_one_is_very_long(self):
        text_line = [
            bout(0, 300, 405, 130),
            bout(1, 420, 410, 160),
            bout(2, 560, 399, 140),
            bout(3, 700, 407, 150),
            bout(4, 820, 403, 120),
            bout(5, 840, 406, 135),
        ]
        off_line = bout(6, 610, 650, 3000)
        retained, diagnostic = estimate_text_line(
            text_line + [off_line], 260, 850
        )

        self.assertEqual(diagnostic.status, "ok")
        self.assertEqual([b.sequence for b in retained], [0, 1, 2, 3, 4, 5])

    def test_rejects_trial_without_enough_horizontal_evidence(self):
        clustered = [
            bout(0, 500, 380),
            bout(1, 510, 385),
            bout(2, 520, 375),
        ]
        retained, diagnostic = estimate_text_line(clustered, 260, 850)

        self.assertEqual(retained, [])
        self.assertEqual(diagnostic.status, "rejected")
        self.assertIn("insufficient", diagnostic.quality_flags)

    def test_ignores_fixations_far_outside_sentence_x_range(self):
        text_line = [
            bout(0, 300, 330),
            bout(1, 430, 335),
            bout(2, 570, 328),
            bout(3, 720, 332),
            bout(4, 760, 329),
            bout(5, 790, 334),
        ]
        far_away = [bout(6, 1100, 600, 1000), bout(7, 1150, 610, 1000)]
        retained, diagnostic = estimate_text_line(
            text_line + far_away, 260, 800
        )

        self.assertEqual(diagnostic.status, "ok")
        self.assertEqual(len(retained), 6)
        self.assertEqual(diagnostic.n_candidate_bouts, 6)

    def test_recovers_tilted_line(self):
        xs = [290, 390, 490, 590, 690, 790, 830]
        tilted = [
            bout(i, x, 360 + 0.14 * (x - 560)) for i, x in enumerate(xs)
        ]
        retained, diagnostic = estimate_text_line(tilted, 260, 850)

        self.assertEqual(diagnostic.status, "ok")
        self.assertEqual(len(retained), len(tilted))
        self.assertAlmostEqual(diagnostic.beta, 0.14, delta=0.03)

    def test_rejects_two_equally_supported_vertical_bands(self):
        xs = [290, 390, 490, 590, 690, 790]
        first = [bout(i, x, 300) for i, x in enumerate(xs)]
        second = [bout(i + 6, x, 420) for i, x in enumerate(xs)]
        retained, diagnostic = estimate_text_line(first + second, 260, 850)

        self.assertEqual(retained, [])
        self.assertEqual(diagnostic.reason, "ambiguous_competing_line")

    def test_coordinate_missing_bout_is_preserved_as_unknown(self):
        content = "\n".join(
            [
                "TimeStamp,X,Y,Event",
                "9:00:00.0000,,,fixation",
                "9:00:00.0300,,,fixation",
                "9:00:00.0310,,,saccade",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.csv"
            path.write_text(content, encoding="utf-8")
            bouts = read_fixation_bouts(path)

        self.assertEqual(len(bouts), 1)
        self.assertIsNone(bouts[0].mean_x)
        self.assertIsNone(bouts[0].mean_y)
        self.assertGreaterEqual(bouts[0].duration_ms, 30)

    def test_interword_space_is_divided_at_gap_midpoint(self):
        ranges = [(0, 100, "long"), (120, 130, "x")]

        self.assertEqual(x_to_word_index(109, ranges), 0)
        self.assertEqual(x_to_word_index(111, ranges), 1)
        self.assertEqual(x_to_word_index(-30, ranges), 0)
        self.assertEqual(x_to_word_index(-31, ranges), -1)
        self.assertEqual(x_to_word_index(160, ranges), 1)
        self.assertEqual(x_to_word_index(161, ranges), -1)

    def test_read_prior_keeps_fixed_slope_and_consistent_band(self):
        ranges = self.ranges()
        x_center = (ranges[0][0] + ranges[-1][1]) / 2
        prior = LineModel(
            alpha=400,
            beta=0.10,
            x_center=x_center,
            half_width=24,
            fit_source="independent",
        )
        xs = [300, 430, 670, 810]
        translation = [
            bout(i, x, 350 + prior.beta * (x - x_center) + noise)
            for i, (x, noise) in enumerate(zip(xs, [-4, 3, -2, 5]))
        ]

        retained, model, diagnostic = fit_text_line(
            translation, ranges, stage="translate", prior_model=prior
        )

        self.assertEqual(diagnostic.status, "ok")
        self.assertEqual(model.fit_source, "read_prior")
        self.assertAlmostEqual(model.beta, prior.beta)
        self.assertLessEqual(abs(model.alpha - prior.alpha), 80)
        self.assertEqual(diagnostic.independent_failure_reason,
                         "fewer_than_six_line_bouts")
        self.assertIn("read_prior_fallback", diagnostic.review_flags)
        residuals = [b.mean_y - model.predict(b.mean_x) for b in retained]
        centre = median(residuals)
        mad = median(abs(value - centre) for value in residuals)
        expected_width = max(24, min(45, 2.5 * 1.4826 * mad))
        self.assertAlmostEqual(model.half_width, expected_width, places=6)

    def test_read_prior_rejects_equally_supported_competing_band(self):
        ranges = self.ranges()
        x_center = (ranges[0][0] + ranges[-1][1]) / 2
        prior = LineModel(
            alpha=300,
            beta=0,
            x_center=x_center,
            half_width=24,
            fit_source="independent",
        )
        xs = [300, 430, 670, 810]
        first = [bout(i, x, 300) for i, x in enumerate(xs)]
        second = [bout(i + 4, x, 420) for i, x in enumerate(xs)]

        retained, model, diagnostic = fit_text_line(
            first + second, ranges, stage="translate", prior_model=prior
        )

        self.assertEqual(retained, [])
        self.assertIsNone(model)
        self.assertEqual(diagnostic.reason, "ambiguous_prior_competing_line")
        self.assertEqual(diagnostic.independent_failure_reason,
                         "fewer_than_six_line_bouts")

    def test_read_prior_resolves_membership_cycle_deterministically(self):
        # These residuals produce a two-state adaptive-width cycle: the wider
        # subset band includes bout 5, while recomputing MAD after inclusion
        # narrows the band enough to exclude it again.
        residuals = [
            -22.2097, -25.0528, 9.4683, 24.5686, -2.0355, 44.7674,
            14.2437, -7.0388, 12.47, 0.0, 4.8054, -20.1038,
        ]
        candidates = [
            bout(i, 280 + i * 30, 211.5981 + residual)
            for i, residual in enumerate(residuals)
        ]
        prior = LineModel(
            alpha=200,
            beta=0,
            x_center=450,
            half_width=24,
            fit_source="independent",
        )
        seed = _CandidateLine(
            alpha=211.5981,
            beta=0,
            inlier_sequences=frozenset(i for i in range(12) if i != 5),
            score=1,
        )

        model, retained, _, cycle_resolved = _fixed_slope_refit(
            candidates, seed, prior
        )

        self.assertTrue(cycle_resolved)
        self.assertEqual(len(retained), 11)
        self.assertTrue(
            all(
                abs(item.mean_y - model.predict(item.mean_x))
                <= model.half_width
                for item in retained
            )
        )
        retained_sequences = {item.sequence for item in retained}
        self.assertNotIn(5, retained_sequences)
        self.assertGreater(
            abs(candidates[5].mean_y - model.predict(candidates[5].mean_x)),
            model.half_width,
        )

    def test_four_word_sentence_can_pass_with_three_word_aois(self):
        ranges = [
            (260 + i * 100, 340 + i * 100, str(i)) for i in range(4)
        ]
        # The third word is skipped, but the scanpath has strong support and
        # horizontal coverage across the other three words.
        xs = [270, 320, 370, 420, 570, 630]
        scanpath = [bout(i, x, 350) for i, x in enumerate(xs)]

        retained, model, diagnostic = fit_text_line(
            scanpath, ranges, stage="read"
        )

        self.assertEqual(diagnostic.status, "ok")
        self.assertIsNotNone(model)
        self.assertEqual(len(retained), 6)
        self.assertEqual(diagnostic.n_word_aois, 3)

    def test_empty_translation_retains_structural_failure_reason(self):
        ranges = self.ranges()
        prior = LineModel(
            alpha=300,
            beta=0,
            x_center=550,
            half_width=24,
            fit_source="independent",
        )

        retained, model, diagnostic = fit_text_line(
            [], ranges, stage="translate", prior_model=prior
        )

        self.assertEqual(retained, [])
        self.assertIsNone(model)
        self.assertEqual(diagnostic.reason, "no_fixation_bouts")
        self.assertEqual(
            diagnostic.independent_failure_reason, "no_fixation_bouts"
        )

    def test_minimum_prior_word_coverage_is_review_flagged(self):
        ranges = self.ranges()
        prior = LineModel(
            alpha=350,
            beta=0,
            x_center=550,
            half_width=24,
            fit_source="independent",
        )
        scanpath = [bout(0, 270, 350), bout(1, 330, 352), bout(2, 430, 348)]

        _, model, diagnostic = fit_text_line(
            scanpath, ranges, stage="translate", prior_model=prior
        )

        self.assertEqual(diagnostic.status, "ok")
        self.assertIsNotNone(model)
        self.assertEqual(diagnostic.n_x_bins, 3)
        self.assertEqual(diagnostic.n_word_aois, 2)
        self.assertIn(
            "limited_prior_horizontal_coverage", diagnostic.review_flags
        )


if __name__ == "__main__":
    unittest.main()
