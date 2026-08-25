"""Build leakage-safe train/validation/test manifests for RAID.

This module deliberately does not tokenize text or train a model.  Its job is
to turn raw RAID rows into reproducible records with:

* a binary human/AI label;
* a lineage-aware group identifier; and
* a split assignment that never separates related records.

The implementation uses only the Python standard library so the data contract
can be tested before installing Transformer training dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}


def _optional_string(value: object) -> Optional[str]:
    """Convert empty CSV values to ``None`` while preserving real strings."""

    if value is None:
        return None
    text = str(value)
    return text if text else None


def _required_string(value: object, field_name: str, row_number: int) -> str:
    text = _optional_string(value)
    if text is None:
        raise ValueError(f"Row {row_number} is missing required field '{field_name}'")
    return text


def label_from_model(model: str) -> int:
    """Return the project label: human is 0, every other model is AI (1)."""

    normalized = model.strip().lower()
    if not normalized:
        raise ValueError("model cannot be empty when creating a label")
    return 0 if normalized == "human" else 1


@dataclass
class ManifestRecord:
    """A raw RAID example plus fields created by the manifest builder."""

    record_id: str
    text: str
    label: int
    source_id: Optional[str]
    adv_source_id: Optional[str]
    model: str
    decoding: Optional[str]
    repetition_penalty: Optional[str]
    attack: str
    domain: str
    title: str
    prompt: Optional[str]
    group_id: str = ""
    split: str = ""
    text_length: int = 0
    text_hash: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def records_from_rows(rows: Iterable[Mapping[str, object]]) -> Tuple[List[ManifestRecord], Dict[str, int]]:
    """Normalize raw RAID dictionaries and remove exact duplicate texts.

    Duplicate records with the same label are reduced deterministically to the
    record with the lexicographically smallest ID.  Duplicate text with
    conflicting labels is rejected because silently choosing a label would
    hide a data-quality problem.
    """

    candidates: List[ManifestRecord] = []
    skipped_empty = 0

    for row_number, row in enumerate(rows, start=2):
        record_id = _required_string(row.get("id"), "id", row_number)
        model = _required_string(row.get("model"), "model", row_number)
        raw_text = row.get("generation")
        if raw_text is None:
            raw_text = row.get("text")
        text = "" if raw_text is None else str(raw_text)

        if not text.strip():
            skipped_empty += 1
            continue

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        candidates.append(
            ManifestRecord(
                record_id=record_id,
                text=text,
                label=label_from_model(model),
                source_id=_optional_string(row.get("source_id")),
                adv_source_id=_optional_string(row.get("adv_source_id")),
                model=model,
                decoding=_optional_string(row.get("decoding")),
                repetition_penalty=_optional_string(row.get("repetition_penalty")),
                attack=_optional_string(row.get("attack")) or "none",
                domain=_optional_string(row.get("domain")) or "unknown",
                title=str(row.get("title") or ""),
                prompt=_optional_string(row.get("prompt")),
                text_length=len(text),
                text_hash=text_hash,
            )
        )

    # Sorting makes duplicate selection independent of input order.
    candidates.sort(key=lambda record: record.record_id)
    kept: List[ManifestRecord] = []
    first_by_hash: Dict[str, ManifestRecord] = {}
    duplicate_texts = 0

    for record in candidates:
        previous = first_by_hash.get(record.text_hash)
        if previous is None:
            first_by_hash[record.text_hash] = record
            kept.append(record)
            continue

        if previous.label != record.label:
            raise ValueError(
                "Exact duplicate text has conflicting labels: "
                f"{previous.record_id}={previous.label}, {record.record_id}={record.label}"
            )
        duplicate_texts += 1

    stats = {
        "rows_kept": len(kept),
        "empty_rows_removed": skipped_empty,
        "duplicate_texts_removed": duplicate_texts,
    }
    return kept, stats


class _UnionFind:
    """Small disjoint-set structure for connecting source lineage IDs."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Lexicographic roots make the result independent of row order.
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def assign_group_ids(records: Sequence[ManifestRecord]) -> int:
    """Assign stable lineage groups and return the number of groups.

    A source ID and an adversarial-source ID are treated as links.  Connected
    links are intentionally merged: if one record connects source X to
    adversarial source Y, all records in either lineage remain together.
    """

    union_find = _UnionFind()
    record_links: Dict[int, List[str]] = defaultdict(list)

    for index, record in enumerate(records):
        links = []
        if record.source_id:
            links.append(f"source:{record.source_id}")
        if record.adv_source_id:
            links.append(f"adv:{record.adv_source_id}")

        if not links:
            links = [f"record:{record.record_id}"]

        record_links[index] = links
        for link in links:
            union_find.add(link)
        for link in links[1:]:
            union_find.union(links[0], link)

    component_members: Dict[str, List[str]] = defaultdict(list)
    for links in record_links.values():
        root = union_find.find(links[0])
        component_members[root].extend(links)

    stable_group_by_root: Dict[str, str] = {}
    for root, members in component_members.items():
        representative = min(set(members))
        stable_group_by_root[root] = "group-" + hashlib.sha256(
            representative.encode("utf-8")
        ).hexdigest()[:16]

    for index, record in enumerate(records):
        root = union_find.find(record_links[index][0])
        record.group_id = stable_group_by_root[root]

    return len(stable_group_by_root)


def _split_cost(
    totals: Mapping[str, int],
    label_totals: Mapping[str, Mapping[int, int]],
    group_size: int,
    group_labels: Mapping[int, int],
    target_total: float,
    target_labels: Mapping[int, float],
) -> float:
    """Score how well adding a group preserves target size and label ratios."""

    new_total = totals["current"] + group_size
    total_error = abs(new_total - target_total) / max(target_total, 1.0)
    label_error = 0.0
    for label in (0, 1):
        new_count = label_totals["current"].get(label, 0) + group_labels.get(label, 0)
        label_error += abs(new_count - target_labels[label]) / max(target_labels[label], 1.0)

    # Label balance is more important than exact row counts for the pilot.
    return total_error + 2.0 * label_error


def assign_splits(
    records: Sequence[ManifestRecord],
    seed: int = 42,
    ratios: Optional[Mapping[str, float]] = None,
) -> Dict[str, int]:
    """Assign whole groups to train/validation/test deterministically."""

    split_ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    if set(split_ratios) != set(SPLITS):
        raise ValueError(f"ratios must contain exactly {SPLITS}")
    if abs(sum(split_ratios.values()) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1")

    grouped: Dict[str, List[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)

    rng = random.Random(seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: -len(item[1]))

    if len(group_items) < len(SPLITS):
        raise ValueError(
            f"need at least {len(SPLITS)} lineage groups to create the requested splits; "
            f"found {len(group_items)}"
        )

    total_rows = len(records)
    total_labels = Counter(record.label for record in records)
    assigned_rows = {split: {"current": 0} for split in SPLITS}
    assigned_labels = {split: {"current": Counter()} for split in SPLITS}
    assignments: Dict[str, int] = {}

    # Seed every partition with one whole group.  Without this step a greedy
    # size optimizer can legitimately decide that a tiny validation/test target
    # is best represented by zero rows on a small smoke-test dataset.
    remaining = list(group_items)
    for split in SPLITS:
        target_total = total_rows * split_ratios[split]
        target_labels = {label: total_labels[label] * split_ratios[split] for label in (0, 1)}

        def seed_cost(item: Tuple[str, List[ManifestRecord]]) -> Tuple[float, str]:
            group_id, group_records = item
            counts = Counter(record.label for record in group_records)
            size_error = abs(len(group_records) - target_total) / max(target_total, 1.0)
            label_error = sum(
                abs(counts.get(label, 0) - target_labels[label]) / max(target_labels[label], 1.0)
                for label in (0, 1)
            )
            return size_error + label_error, group_id

        selected_group_id, selected_records = min(remaining, key=seed_cost)
        remaining = [item for item in remaining if item[0] != selected_group_id]
        selected_labels = Counter(record.label for record in selected_records)
        assignments[selected_group_id] = split
        assigned_rows[split]["current"] += len(selected_records)
        assigned_labels[split]["current"].update(selected_labels)

    for group_id, group_records in remaining:
        group_labels = Counter(record.label for record in group_records)
        choices: List[Tuple[float, str]] = []
        for split in SPLITS:
            cost = _split_cost(
                assigned_rows[split],
                assigned_labels[split],
                len(group_records),
                group_labels,
                total_rows * split_ratios[split],
                {label: total_labels[label] * split_ratios[split] for label in (0, 1)},
            )
            current_labels = assigned_labels[split]["current"]
            missing_labels = [label for label in (0, 1) if current_labels.get(label, 0) == 0]
            filled_missing = sum(1 for label in missing_labels if group_labels.get(label, 0) > 0)
            # Encourage a group that supplies a missing class, without
            # allowing this preference to split a lineage group.
            cost -= 3.0 * filled_missing
            choices.append((cost, split))

        _, selected_split = min(choices)
        assignments[group_id] = selected_split
        assigned_rows[selected_split]["current"] += len(group_records)
        assigned_labels[selected_split]["current"].update(group_labels)

    for record in records:
        record.split = assignments[record.group_id]

    return assignments


def validate_manifest(records: Sequence[ManifestRecord], require_both_classes: bool = True) -> None:
    """Fail loudly if a manifest violates its leakage and label invariants."""

    if not records:
        raise ValueError("manifest contains no records")

    group_splits: Dict[str, str] = {}
    source_splits: Dict[str, str] = {}
    adv_source_splits: Dict[str, str] = {}
    text_splits: Dict[str, str] = {}

    for record in records:
        if record.label not in (0, 1):
            raise ValueError(f"invalid label for {record.record_id}: {record.label}")
        if record.split not in SPLITS:
            raise ValueError(f"invalid split for {record.record_id}: {record.split}")

        previous_group_split = group_splits.setdefault(record.group_id, record.split)
        if previous_group_split != record.split:
            raise ValueError(f"group leakage detected for {record.group_id}")

        for value, seen, field_name in (
            (record.source_id, source_splits, "source_id"),
            (record.adv_source_id, adv_source_splits, "adv_source_id"),
            (record.text_hash, text_splits, "text"),
        ):
            if value is None:
                continue
            previous_split = seen.setdefault(value, record.split)
            if previous_split != record.split:
                raise ValueError(f"{field_name} leakage detected for {value}")

    if require_both_classes:
        for split in SPLITS:
            labels = {record.label for record in records if record.split == split}
            if labels != {0, 1}:
                raise ValueError(f"split '{split}' does not contain both labels: {labels}")


def summarize_manifest(records: Sequence[ManifestRecord], preprocessing_stats: Optional[Mapping[str, int]] = None) -> Dict[str, object]:
    """Return JSON-serializable distributions for human inspection."""

    summary: Dict[str, object] = {
        "records": len(records),
        "groups": len({record.group_id for record in records}),
        "splits": {},
        "preprocessing": dict(preprocessing_stats or {}),
    }

    split_summary: Dict[str, object] = {}
    for split in SPLITS:
        subset = [record for record in records if record.split == split]
        split_summary[split] = {
            "rows": len(subset),
            "labels": dict(Counter(str(record.label) for record in subset)),
            "models": dict(Counter(record.model for record in subset)),
            "domains": dict(Counter(record.domain for record in subset)),
            "attacks": dict(Counter(record.attack for record in subset)),
        }
    summary["splits"] = split_summary
    return summary


def read_raid_csv(path: Path, max_rows: Optional[int] = None) -> Iterable[Mapping[str, object]]:
    """Stream raw RAID rows from a CSV file."""

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {"id", "model"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            yield row


def write_manifests(
    records: Sequence[ManifestRecord],
    output_dir: Path,
    preprocessing_stats: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    """Write one JSONL file per split plus a compact summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record.split == split:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    summary = summarize_manifest(records, preprocessing_stats)
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def build_manifest(
    input_path: Path,
    output_dir: Path,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """Build and write a complete manifest from a local labeled RAID CSV."""

    rows = read_raid_csv(input_path, max_rows=max_rows)
    records, preprocessing_stats = records_from_rows(rows)
    group_count = assign_group_ids(records)
    assign_splits(records, seed=seed)
    validate_manifest(records)

    summary = write_manifests(records, output_dir, preprocessing_stats)
    summary["seed"] = seed
    summary["input"] = str(input_path)
    summary["group_count"] = group_count
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Labeled RAID CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Manifest output directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional bounded smoke-test size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_manifest(args.input, args.output_dir, max_rows=args.max_rows, seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
