"""Join the selected RAID pilot assignments with the original RAID text.

The pilot dataset contains record IDs and labels but intentionally does not
contain raw text.  This job reads those IDs from a Kaggle Dataset, streams the
official ``liamdugan/raid`` training split, and writes only the 10,000 matching
records.  It validates IDs, labels, metadata, and text hashes before declaring
the join successful.

Environment variables:

    RAID_HF_DATASET=liamdugan/raid
    RAID_SPLIT=train
    PILOT_INPUT_DIR=/kaggle/input
    PILOT_OUTPUT_DIR=/kaggle/working
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple


PILOT_COLUMNS = {
    "record_id",
    "group_id",
    "label",
    "split",
    "model",
    "domain",
    "attack",
}
COMPARE_COLUMNS = (
    "model",
    "domain",
    "attack",
    "decoding",
    "repetition_penalty",
)


def find_pilot_file(input_dir: Path, filename: str) -> Path:
    """Find a pilot file below a Kaggle input directory."""

    matches = sorted(input_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"could not find {filename} below {input_dir}")
    if len(matches) > 1:
        raise ValueError(f"found multiple {filename} files below {input_dir}: {matches}")
    return matches[0]


def read_pilot_assignments(path: Path) -> Dict[str, Dict[str, str]]:
    """Read pilot metadata keyed by record ID and reject duplicate IDs."""

    assignments: Dict[str, Dict[str, str]] = {}
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = PILOT_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"pilot assignments missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            record_id = row.get("record_id")
            if not record_id:
                raise ValueError(f"pilot assignments row {row_number} has no record_id")
            if record_id in assignments:
                raise ValueError(f"duplicate pilot record_id: {record_id}")
            if row.get("split") != "train":
                raise ValueError(f"pilot record {record_id} is not from split=train")
            if row.get("label") not in {"0", "1"}:
                raise ValueError(f"pilot record {record_id} has an invalid label")
            assignments[record_id] = row
    if not assignments:
        raise ValueError("pilot assignments are empty")
    return assignments


def read_pilot_ids(path: Path) -> Set[str]:
    """Read and validate the one-record-ID-per-line pilot file."""

    ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not ids:
        raise ValueError("pilot record ID file is empty")
    return ids


def _normalized(value: object) -> str:
    return "" if value is None else str(value)


def _raw_text(row: Mapping[str, object]) -> str:
    value = row.get("generation")
    if value is None:
        value = row.get("text")
    return "" if value is None else str(value)


def _raw_model(row: Mapping[str, object]) -> str:
    return _normalized(row.get("model"))


def join_stream(
    raw_rows: Iterable[Mapping[str, object]],
    assignments: Mapping[str, Mapping[str, str]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Join matching raw rows and validate them against pilot metadata."""

    joined: List[Dict[str, object]] = []
    seen: Set[str] = set()
    metadata_mismatches: Counter = Counter()
    empty_text = 0

    for raw_row in raw_rows:
        record_id = _normalized(raw_row.get("id"))
        if record_id not in assignments:
            continue
        if record_id in seen:
            raise ValueError(f"raw RAID stream contains duplicate pilot ID: {record_id}")
        seen.add(record_id)

        assignment = assignments[record_id]
        text = _raw_text(raw_row)
        if not text.strip():
            empty_text += 1
        expected_values = {
            "model": assignment.get("model"),
            "domain": assignment.get("domain"),
            "attack": assignment.get("attack"),
            "decoding": assignment.get("decoding"),
            "repetition_penalty": assignment.get("repetition_penalty"),
        }
        for field in COMPARE_COLUMNS:
            if _normalized(raw_row.get(field)) != _normalized(expected_values[field]):
                metadata_mismatches[field] += 1

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        expected_hash = assignment.get("text_hash")
        if expected_hash and text_hash != expected_hash:
            metadata_mismatches["text_hash"] += 1

        expected_label = int(assignment["label"])
        actual_label = 0 if _raw_model(raw_row).strip().lower() == "human" else 1
        if actual_label != expected_label:
            metadata_mismatches["label"] += 1

        joined.append(
            {
                "record_id": record_id,
                "text": text,
                "label": expected_label,
                "group_id": assignment["group_id"],
                "split": assignment["split"],
                "source_id": raw_row.get("source_id"),
                "adv_source_id": raw_row.get("adv_source_id"),
                "model": raw_row.get("model"),
                "domain": raw_row.get("domain"),
                "attack": raw_row.get("attack") or "none",
                "decoding": raw_row.get("decoding"),
                "repetition_penalty": raw_row.get("repetition_penalty"),
                "text_length": len(text),
                "text_hash": text_hash,
            }
        )

    missing_ids = sorted(set(assignments) - seen)
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} pilot IDs were not found; examples: {missing_ids[:5]}"
        )
    if metadata_mismatches:
        raise ValueError(f"pilot/raw metadata mismatches: {dict(metadata_mismatches)}")

    summary = {
        "requested_rows": len(assignments),
        "joined_rows": len(joined),
        "labels": dict(Counter(str(row["label"]) for row in joined)),
        "groups": len({row["group_id"] for row in joined}),
        "empty_text_rows": empty_text,
        "metadata_mismatches": dict(metadata_mismatches),
    }
    return joined, summary


def write_outputs(rows: List[Mapping[str, object]], summary: Mapping[str, object], output_dir: Path) -> None:
    """Write joined JSONL and summary atomically after validation succeeds."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raid_pilot_joined.jsonl.gz"
    temporary_path = output_dir / ".raid_pilot_joined.jsonl.gz.tmp"
    with gzip.open(temporary_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    with (output_dir / "raid_pilot_join_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def iter_huggingface_rows(dataset_name: str, split: str) -> Iterable[Mapping[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Kaggle environment needs the 'datasets' package") from error
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    yield from dataset


def build_join(
    pilot_input_dir: Path,
    output_dir: Path,
    dataset_name: str = "liamdugan/raid",
    split: str = "train",
) -> Dict[str, object]:
    """Build the validated pilot/text join."""

    try:
        assignments_path = find_pilot_file(pilot_input_dir, "pilot_assignments.csv.gz")
    except FileNotFoundError:
        assignments_path = find_pilot_file(pilot_input_dir, "pilot_assignments.csv")
    ids_path = find_pilot_file(pilot_input_dir, "pilot_record_ids.txt")
    assignments = read_pilot_assignments(assignments_path)
    ids = read_pilot_ids(ids_path)
    if ids != set(assignments):
        raise ValueError("pilot ID file and pilot assignment IDs differ")

    joined, summary = join_stream(
        iter_huggingface_rows(dataset_name, split),
        assignments,
    )
    summary.update(
        {
            "dataset": dataset_name,
            "split": split,
            "pilot_rows": len(assignments),
        }
    )
    write_outputs(joined, summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-input-dir",
        type=Path,
        default=Path(os.getenv("PILOT_INPUT_DIR", "/kaggle/input")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("PILOT_OUTPUT_DIR", "/kaggle/working")),
    )
    parser.add_argument("--dataset", default=os.getenv("RAID_HF_DATASET", "liamdugan/raid"))
    parser.add_argument("--split", default=os.getenv("RAID_SPLIT", "train"))
    args = parser.parse_args()
    summary = build_join(args.pilot_input_dir, args.output_dir, args.dataset, args.split)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
