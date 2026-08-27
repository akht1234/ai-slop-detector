"""Tests for tokenizer-validation data and shape checks."""

import tempfile
import unittest
from pathlib import Path

from src.training.tokenizer_validation import (
    read_jsonl,
    summarize_lengths,
    validate_tokenizer,
)


class FakeTokenizer:
    """Small tokenizer double that behaves like the fields we use from HF."""

    def __init__(self, limit=6):
        self.limit = limit

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        truncation,
        padding,
        max_length=None,
        return_attention_mask=False,
        return_special_tokens_mask=False,
    ):
        tokens = text.split()
        ids = [10 + index for index, _ in enumerate(tokens)]
        if add_special_tokens:
            ids = [0] + ids + [2]
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        special = [1 if value in (0, 2) else 0 for value in ids]
        if padding == "max_length":
            ids += [1] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))
            special += [1] * (max_length - len(special))
        result = {"input_ids": ids}
        if return_attention_mask:
            result["attention_mask"] = mask
        if return_special_tokens_mask:
            result["special_tokens_mask"] = special
        return result

    def convert_ids_to_tokens(self, ids):
        return [f"token-{value}" for value in ids]


class TokenizerValidationTests(unittest.TestCase):
    def test_summarize_lengths(self):
        summary = summarize_lengths([1, 2, 3, 4, 5])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 3.0)
        self.assertEqual(summary["p90"], 4.6)

    def test_validation_reports_lengths_labels_and_truncation(self):
        rows = [
            {"record_id": "h1", "text": "one two", "label": 0, "model": "human", "attack": "none"},
            {"record_id": "a1", "text": "one two three four five six", "label": 1, "model": "gpt2", "attack": "none"},
        ]
        report = validate_tokenizer(rows, FakeTokenizer(), max_length=6, progress_every=1)
        self.assertEqual(report["rows_seen"], 2)
        self.assertEqual(report["labels"], {"0": 1, "1": 1})
        self.assertEqual(report["truncated_rows"], 1)
        self.assertEqual(report["truncation_rate"], 0.5)
        self.assertEqual(len(report["examples"][0]["tokens"]), 4)

    def test_validation_rejects_invalid_progress_interval(self):
        rows = [{"record_id": "x", "text": "text", "label": 0}]
        with self.assertRaisesRegex(ValueError, "progress_every"):
            validate_tokenizer(rows, FakeTokenizer(), progress_every=0)

    def test_validation_rejects_bad_label(self):
        rows = [{"record_id": "x", "text": "text", "label": 2}]
        with self.assertRaisesRegex(ValueError, "invalid label"):
            validate_tokenizer(rows, FakeTokenizer(), max_length=6)

    def test_read_jsonl_supports_jsonl_gz(self):
        import gzip
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"record_id": "r1", "text": "x", "label": 0}) + "\n")
            self.assertEqual(list(read_jsonl(path))[0]["record_id"], "r1")


if __name__ == "__main__":
    unittest.main()
