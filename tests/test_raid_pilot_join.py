"""Tests for the RAID pilot/text join validation helpers."""

import hashlib
import importlib.util
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "kaggle" / "raid_pilot_join" / "raid_pilot_join.py"
_SPEC = importlib.util.spec_from_file_location("raid_pilot_join", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
join_stream = _MODULE.join_stream


class RAIDPilotJoinTests(unittest.TestCase):
    def test_join_validates_labels_metadata_and_hashes(self):
        human_text = "A human passage."
        ai_text = "An AI passage."
        assignments = {
            "h1": {
                "record_id": "h1",
                "group_id": "g1",
                "label": "0",
                "split": "train",
                "model": "human",
                "domain": "news",
                "attack": "none",
                "decoding": "greedy",
                "repetition_penalty": "1.0",
                "text_hash": hashlib.sha256(human_text.encode()).hexdigest(),
            },
            "a1": {
                "record_id": "a1",
                "group_id": "g1",
                "label": "1",
                "split": "train",
                "model": "gpt2",
                "domain": "news",
                "attack": "none",
                "decoding": "greedy",
                "repetition_penalty": "1.0",
                "text_hash": hashlib.sha256(ai_text.encode()).hexdigest(),
            },
        }
        rows, summary = join_stream(
            [
                {
                    "id": "h1",
                    "generation": human_text,
                    "model": "human",
                    "domain": "news",
                    "attack": "none",
                    "decoding": "greedy",
                    "repetition_penalty": "1.0",
                },
                {
                    "id": "a1",
                    "generation": ai_text,
                    "model": "gpt2",
                    "domain": "news",
                    "attack": "none",
                    "decoding": "greedy",
                    "repetition_penalty": "1.0",
                },
            ],
            assignments,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["labels"], {"0": 1, "1": 1})
        self.assertEqual(summary["metadata_mismatches"], {})

    def test_join_rejects_missing_ids(self):
        assignments = {
            "missing": {
                "record_id": "missing",
                "group_id": "g1",
                "label": "0",
                "split": "train",
                "model": "human",
                "domain": "news",
                "attack": "none",
            }
        }
        with self.assertRaisesRegex(ValueError, "not found"):
            join_stream([], assignments)


if __name__ == "__main__":
    unittest.main()
