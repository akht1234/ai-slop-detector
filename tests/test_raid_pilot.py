"""Tests for group-safe RAID pilot selection."""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.data.raid_pilot import build_pilot, select_groups, summarize_groups


def assignment_rows():
    return [
        {"record_id": "h1", "group_id": "g1", "label": "0", "split": "train", "model": "human", "domain": "news", "attack": "none"},
        {"record_id": "a1", "group_id": "g1", "label": "1", "split": "train", "model": "mistral", "domain": "news", "attack": "none"},
        {"record_id": "h2", "group_id": "g2", "label": "0", "split": "train", "model": "human", "domain": "books", "attack": "none"},
        {"record_id": "a2", "group_id": "g2", "label": "1", "split": "train", "model": "gpt2", "domain": "books", "attack": "whitespace"},
        {"record_id": "h3", "group_id": "g3", "label": "0", "split": "train", "model": "human", "domain": "reddit", "attack": "none"},
        {"record_id": "a3", "group_id": "g3", "label": "1", "split": "train", "model": "mpt", "domain": "reddit", "attack": "synonym"},
    ]


class RAIDPilotTests(unittest.TestCase):
    def test_summarize_groups(self):
        groups = summarize_groups(assignment_rows())
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups["g1"].labels, {0: 1, 1: 1})

    def test_selection_uses_training_groups_for_safety(self):
        groups = summarize_groups(assignment_rows())
        selected, summary = select_groups(groups, target_rows=4, seed=42)
        self.assertEqual(len(selected), 2)
        self.assertEqual(summary["selected_human_capacity"], 2)

    def test_build_pilot_writes_ids_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir = Path(directory) / "manifest"
            output_dir = Path(directory) / "pilot"
            manifest_dir.mkdir()
            fieldnames = list(assignment_rows()[0])

            def write_shard(path, rows):
                with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

            write_shard(
                manifest_dir / "record_assignments_train_00.csv.gz",
                assignment_rows(),
            )
            validation_row = dict(assignment_rows()[0])
            validation_row.update(record_id="v1", group_id="validation-group", split="validation")
            test_row = dict(assignment_rows()[1])
            test_row.update(record_id="t1", group_id="test-group", split="test")
            write_shard(
                manifest_dir / "record_assignments_validation_00.csv.gz",
                [validation_row],
            )
            write_shard(
                manifest_dir / "record_assignments_test_00.csv.gz",
                [test_row],
            )

            summary = build_pilot(manifest_dir, output_dir, target_rows=4)
            self.assertEqual(summary["rows_written"], 4)
            self.assertEqual(summary["partition_overlap_checks"], {"validation": 0, "test": 0})
            self.assertEqual(len(output_dir.joinpath("pilot_record_ids.txt").read_text().splitlines()), 4)
            loaded = json.loads(output_dir.joinpath("pilot_summary.json").read_text())
            self.assertEqual(loaded["groups_written"], 2)
            self.assertTrue(output_dir.joinpath("pilot_assignments.csv.gz").exists())


if __name__ == "__main__":
    unittest.main()
