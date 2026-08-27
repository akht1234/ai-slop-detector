"""Tests for RAID labels, lineage groups, and split leakage checks."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.raid_manifest import (
    assign_group_ids,
    assign_splits,
    records_from_rows,
    validate_manifest,
    write_manifests,
)


def example_rows():
    return [
        {
            "id": "human-1",
            "source_id": "source-1",
            "adv_source_id": "source-1",
            "model": "human",
            "attack": "none",
            "domain": "abstracts",
            "generation": "human source text one",
        },
        {
            "id": "ai-1",
            "source_id": "source-1",
            "adv_source_id": "source-1",
            "model": "mistral",
            "attack": "none",
            "domain": "abstracts",
            "generation": "AI generation one",
        },
        {
            "id": "attack-1",
            "source_id": "source-1",
            "adv_source_id": "adv-1",
            "model": "mistral",
            "attack": "whitespace",
            "domain": "abstracts",
            "generation": "AI generation one attacked",
        },
        {
            "id": "human-2",
            "source_id": "source-2",
            "adv_source_id": "source-2",
            "model": "human",
            "attack": "none",
            "domain": "abstracts",
            "generation": "human source text two",
        },
        {
            "id": "ai-2",
            "source_id": "source-3",
            "adv_source_id": "source-3",
            "model": "gpt2",
            "attack": "none",
            "domain": "abstracts",
            "generation": "AI generation two",
        },
        {
            "id": "human-3",
            "source_id": "source-4",
            "adv_source_id": "source-4",
            "model": "human",
            "attack": "none",
            "domain": "abstracts",
            "generation": "human source text three",
        },
        {
            "id": "ai-3",
            "source_id": "source-5",
            "adv_source_id": "source-5",
            "model": "llama-chat",
            "attack": "upper_lower",
            "domain": "abstracts",
            "generation": "AI generation three",
        },
        {
            "id": "human-4",
            "source_id": "source-6",
            "adv_source_id": "source-6",
            "model": "human",
            "attack": "none",
            "domain": "abstracts",
            "generation": "human source text four",
        },
    ]


class RAIDManifestTests(unittest.TestCase):
    def test_labels_and_exact_duplicates(self):
        rows = example_rows() + [
            {
                "id": "duplicate-later",
                "source_id": "source-7",
                "adv_source_id": "source-7",
                "model": "gpt2",
                "generation": "AI generation three",
            }
        ]
        records, stats = records_from_rows(rows)

        self.assertEqual(stats["duplicate_texts_removed"], 1)
        self.assertEqual({record.label for record in records if record.model == "human"}, {0})
        self.assertEqual({record.label for record in records if record.model != "human"}, {1})

    def test_conflicting_exact_texts_are_quarantined(self):
        rows = example_rows() + [
            {
                "id": "conflict-human",
                "source_id": "source-7",
                "adv_source_id": "source-7",
                "model": "human",
                "generation": "AI generation three",
            }
        ]
        records, stats = records_from_rows(rows)

        self.assertEqual(stats["conflicting_texts_removed"], 1)
        self.assertEqual(stats["conflicting_rows_removed"], 2)
        self.assertNotIn("ai-3", {record.record_id for record in records})
        self.assertNotIn("conflict-human", {record.record_id for record in records})

    def test_lineage_records_share_a_group(self):
        records, _ = records_from_rows(example_rows())
        assign_group_ids(records)

        grouped_ids = {
            record.record_id: record.group_id
            for record in records
            if record.record_id in {"human-1", "ai-1", "attack-1"}
        }
        self.assertEqual(len(set(grouped_ids.values())), 1)

    def test_split_keeps_groups_and_passes_leakage_checks(self):
        records, _ = records_from_rows(example_rows())
        assign_group_ids(records)
        assign_splits(records, seed=7)
        validate_manifest(records)

        by_group = {}
        for record in records:
            by_group.setdefault(record.group_id, record.split)
            self.assertEqual(by_group[record.group_id], record.split)

    def test_writes_jsonl_and_summary(self):
        records, stats = records_from_rows(example_rows())
        assign_group_ids(records)
        assign_splits(records, seed=42)
        validate_manifest(records)

        with tempfile.TemporaryDirectory() as directory:
            summary = write_manifests(records, Path(directory), stats)
            self.assertEqual(summary["records"], len(records))
            for split in ("train", "validation", "test"):
                self.assertTrue((Path(directory) / f"{split}.jsonl").exists())
            loaded = json.loads((Path(directory) / "manifest_summary.json").read_text())
            self.assertEqual(loaded["groups"], summary["groups"])


if __name__ == "__main__":
    unittest.main()
