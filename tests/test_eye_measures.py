import sys
import unittest
from pathlib import Path


EXTRACTION_DIR = (
    Path(__file__).resolve().parents[1] / "extraction" / "eye_tracking"
)
sys.path.insert(0, str(EXTRACTION_DIR))

from extract_eye_measures import compute_measures


class GoPastTimeTests(unittest.TestCase):
    def test_direct_rightward_exit_equals_gaze_duration(self):
        measures = compute_measures([(0, 80.0), (0, 40.0), (1, 100.0)], 3)

        self.assertEqual(measures[0]["gd_ms"], 120.0)
        self.assertEqual(measures[0]["go_past_ms"], 120.0)

    def test_outgoing_regression_is_included_until_rightward_crossing(self):
        bouts = [
            (0, 80.0),
            (1, 100.0),
            (0, 90.0),
            (1, 110.0),
            (2, 120.0),
        ]
        measures = compute_measures(bouts, 3)

        # For word 1, the regression path is word 1 -> word 0 -> word 1;
        # the first fixation on word 2 terminates and is excluded from go-past.
        self.assertEqual(measures[1]["gd_ms"], 100.0)
        self.assertEqual(measures[1]["go_past_ms"], 300.0)

    def test_first_landing_by_regression_in_is_not_genuine_first_pass(self):
        bouts = [(2, 80.0), (1, 100.0), (2, 120.0), (3, 90.0)]
        measures = compute_measures(bouts, 4)

        # Word 1 is first encountered only after word 2 has already been seen.
        self.assertIsNone(measures[1]["go_past_ms"])
        self.assertEqual(measures[1]["first_encounter_status"], "regression")
        # Word 2, by contrast, was encountered in forward first pass; its
        # regression to word 1 is included until the later crossing to word 3.
        self.assertEqual(measures[2]["go_past_ms"], 300.0)
        self.assertEqual(measures[2]["first_encounter_status"], "progressive")

    def test_no_subsequent_rightward_crossing_is_structurally_missing(self):
        bouts = [(0, 80.0), (1, 100.0), (0, 90.0)]
        measures = compute_measures(bouts, 3)

        self.assertEqual(measures[0]["go_past_ms"], 80.0)
        self.assertIsNone(measures[1]["go_past_ms"])

    def test_last_word_go_past_is_structurally_missing(self):
        measures = compute_measures([(0, 80.0), (1, 90.0), (2, 100.0)], 3)

        self.assertIsNone(measures[2]["go_past_ms"])

    def test_off_text_bout_terminates_first_pass(self):
        bouts = [(0, 100.0), (-1, 60.0), (0, 80.0), (1, 90.0)]
        measures = compute_measures(bouts, 2)

        self.assertEqual(measures[0]["gd_ms"], 100.0)
        self.assertEqual(measures[0]["rrt_ms"], 80.0)
        self.assertEqual(measures[0]["reread_occurrence"], 1)
        self.assertEqual(measures[0]["regress_in"], 1)
        self.assertIsNone(measures[0]["go_past_ms"])
        self.assertEqual(
            measures[0]["go_past_status"], "intervening_unmapped_bout"
        )

    def test_unknown_bout_terminates_first_pass(self):
        bouts = [(0, 100.0), (-2, 60.0), (0, 80.0), (1, 90.0)]
        measures = compute_measures(bouts, 2)

        self.assertEqual(measures[0]["gd_ms"], 100.0)
        self.assertEqual(measures[0]["rrt_ms"], 80.0)
        self.assertEqual(measures[0]["reread_occurrence"], 1)
        self.assertIsNone(measures[0]["go_past_ms"])
        self.assertEqual(
            measures[0]["go_past_status"], "intervening_unmapped_bout"
        )


if __name__ == "__main__":
    unittest.main()
