import sys
import unittest
from pathlib import Path


EXTRACTION_DIR = (
    Path(__file__).resolve().parents[1] / "extraction" / "eye_tracking"
)
sys.path.insert(0, str(EXTRACTION_DIR))

from timestamp_utils import ts_to_seconds


class TimestampParsingTests(unittest.TestCase):
    def test_zero_padded_timestamp(self):
        self.assertAlmostEqual(ts_to_seconds("09:18:51.2205"), 33531.2205)

    def test_single_digit_hour(self):
        self.assertAlmostEqual(ts_to_seconds("9:18:51.2205"), 33531.2205)

    def test_single_digit_second(self):
        self.assertAlmostEqual(ts_to_seconds("9:17:3.239000"), 33423.239)

    def test_single_digit_minute(self):
        self.assertAlmostEqual(ts_to_seconds("12:9:28.1865"), 43768.1865)

    def test_trailing_decimal_point(self):
        self.assertAlmostEqual(ts_to_seconds("9:17:3."), 33423.0)

    def test_invalid_timestamp(self):
        with self.assertRaises(ValueError):
            ts_to_seconds("not-a-timestamp")


if __name__ == "__main__":
    unittest.main()
