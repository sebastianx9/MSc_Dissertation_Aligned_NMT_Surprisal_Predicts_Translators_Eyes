import sys
import unittest
from pathlib import Path

import numpy as np


EXTRACTION_DIR = (
    Path(__file__).resolve().parents[1] / "extraction" / "predictors"
)
sys.path.insert(0, str(EXTRACTION_DIR))

from extract_attention_features import (
    lim_normalized_word_features,
    subwords_to_word_map,
)


class LimAttentionFeatureTests(unittest.TestCase):
    def test_uniform_attention_normalizes_every_feature_to_one(self):
        # Three lexical positions plus EOS.  Word 0 has two subwords and word
        # 1 has one, so this also checks segment-size normalization.
        encoder = np.full((1, 1, 4, 4), 0.25)
        cross = np.full((1, 1, 2, 4), 0.25)
        rows = lim_normalized_word_features(
            encoder,
            cross,
            word_map=[0, 0, 1, -1],
            eos_index=3,
            valid_source_positions=[0, 1, 2, 3],
        )

        self.assertEqual([row["word_index"] for row in rows], [0, 1])
        for row in rows:
            for feature in (
                "attn_entropy",
                "attn_context",
                "attn_self",
                "attn_eos",
                "attn_recv",
                "attn_cross",
            ):
                self.assertAlmostEqual(row[feature], 1.0, places=12)

    def test_context_excludes_every_subword_of_the_focal_word(self):
        encoder = np.zeros((1, 1, 4, 4))
        # Both subwords of word 0 attend only to one another.  This is self
        # flow at the word-segment level and must not be counted as context.
        encoder[0, 0, 0, 1] = 1.0
        encoder[0, 0, 1, 0] = 1.0
        encoder[0, 0, 2, 2] = 1.0
        encoder[0, 0, 3, 3] = 1.0
        cross = np.full((1, 1, 1, 4), 0.25)
        rows = lim_normalized_word_features(
            encoder,
            cross,
            word_map=[0, 0, 1, -1],
            eos_index=3,
            valid_source_positions=[0, 1, 2, 3],
        )

        self.assertEqual(rows[0]["attn_context"], 0.0)
        self.assertEqual(rows[0]["attn_recv"], 0.0)
        self.assertEqual(rows[0]["attn_self"], 2.0)

    def test_sentencepiece_mapping_keeps_all_subwords_in_one_word(self):
        tokens = ["▁inter", "national", "ization", "▁works", "</s>"]
        self.assertEqual(subwords_to_word_map(tokens), [0, 0, 0, 1, -1])


if __name__ == "__main__":
    unittest.main()
