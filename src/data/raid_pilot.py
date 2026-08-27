"""Select a small, lineage-safe pilot from RAID assignment shards.

The full RAID manifest contains assignments, not raw text.  This module reads
those assignments and selects a small training pilot for a quick Transformer
pipeline experiment.  It deliberately does not download text, tokenize, or
train a model.

RAID lineage groups are already assigned wholly to train, validation, or test.
For a balanced pilot, we select records only from training groups and may take a
subset of records inside those groups.  This preserves partition safety while
avoiding RAID's natural AI-heavy row ratio.  The selected output contains record
IDs and metadata only; the training job can use those IDs to join the original
RAID text on Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


REQUIRED_COLUMNS = {
    "record_id",
    "group_id",
    "label",
    "split",
    "model",
    "domain",
    "attack",
}
DEFAULT_TARGET_ROWS = 10_000
DEFAULT_HUMAN_FRACTION = 0.5


def assignment_paths(manifest_dir: Path, split: str = "train") -> List[Path]:
    """Return assignment shards for one split in deterministic order."""

    paths = sorted(
        path
        for path in manifest_dir.glob(f"record_assignments_{split}_*")
        if path.suffix in {".gz", ".csv"} or path.name.endswith(".csv.gz")
    )
    if not paths:
        raise FileNotFoundError(
            f"no assignment shards found for split '{split}' in {manifest_dir}"
        )
    return paths


def _open_csv(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_assignment_rows(
    manifest_dir: Path,
    split: str = "train",
) -> Iterable[Mapping[str, str]]:
    """Stream rows from local compressed or uncompressed assignment shards."""

    for path in assignment_paths(manifest_dir, split=split):
        with _open_csv(path) as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                if not row.get("record_id"):
                    raise ValueError(f"{path}:{row_number} has an empty record_id")
                if row.get("split") != split:
                    raise ValueError(
                        f"{path}:{row_number} has split={row.get('split')!r}, expected {split!r}"
                    )
                if row.get("label") not in {"0", "1"}:
                    raise ValueError(f"{path}:{row_number} has an invalid label")
                yield row


@dataclass
class GroupSummary:
    """Counts and coverage metadata for one lineage group."""

    group_id: str
    rows: int = 0
    labels: Counter = field(default_factory=Counter)
    models: Set[str] = field(default_factory=set)
    domains: Set[str] = field(default_factory=set)
    attacks: Set[str] = field(default_factory=set)

    def add(self, row: Mapping[str, str]) -> None:
        self.rows += 1
        self.labels[int(row["label"])] += 1
        self.models.add(row.get("model") or "unknown")
        self.domains.add(row.get("domain") or "unknown")
        self.attacks.add(row.get("attack") or "none")


def summarize_groups(rows: Iterable[Mapping[str, str]]) -> Dict[str, GroupSummary]:
    """Aggregate row counts without retaining the full manifest in memory."""

    groups: Dict[str, GroupSummary] = {}
    for row in rows:
        group_id = row.get("group_id")
        if not group_id:
            raise ValueError("assignment row has an empty group_id")
        group = groups.setdefault(group_id, GroupSummary(group_id=group_id))
        group.add(row)
    if not groups:
        raise ValueError("no assignment rows were found")
    return groups


def verify_group_partition_safety(
    manifest_dir: Path,
    selected_groups: Set[str],
) -> Dict[str, int]:
    """Verify selected training groups do not occur in validation or test."""

    overlaps: Dict[str, int] = {}
    for split in ("validation", "test"):
        split_groups = {
            row["group_id"] for row in read_assignment_rows(manifest_dir, split=split)
        }
        overlap = selected_groups & split_groups
        overlaps[split] = len(overlap)
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(
                f"pilot group leakage detected into {split}: {examples}"
            )
    return overlaps


def _coverage(group: GroupSummary) -> Set[Tuple[str, str]]:
    values: Set[Tuple[str, str]] = set()
    values.update(("model", value) for value in group.models)
    values.update(("domain", value) for value in group.domains)
    values.update(("attack", value) for value in group.attacks)
    return values


def _selection_cost(
    group: GroupSummary,
    current_human_rows: int,
    target_human_rows: int,
    missing_coverage: Set[Tuple[str, str]],
) -> float:
    """Score a group after adding its human records; lower is preferred."""

    human_rows = group.labels.get(0, 0)
    new_human_rows = current_human_rows + human_rows
    human_error = abs(new_human_rows - target_human_rows) / max(target_human_rows, 1)
    overshoot = max(0, new_human_rows - target_human_rows) / max(target_human_rows, 1)
    new_coverage = len(_coverage(group) & missing_coverage)

    # Human count is the primary objective. Coverage breaks ties between
    # otherwise similar groups so the pilot does not become one narrow slice.
    return human_error + 3.0 * overshoot - 0.03 * new_coverage


def select_groups(
    groups: Mapping[str, GroupSummary],
    target_rows: int = DEFAULT_TARGET_ROWS,
    human_fraction: float = DEFAULT_HUMAN_FRACTION,
    seed: int = 42,
) -> Tuple[Set[str], Dict[str, object]]:
    """Select training groups that provide enough human rows for the pilot.

    Groups are used as a safe selection universe, not as the row-level balance
    unit.  The caller later samples records from the selected training groups.
    """

    if target_rows < 2:
        raise ValueError("target_rows must be at least 2")
    if not 0.0 < human_fraction < 1.0:
        raise ValueError("human_fraction must be between 0 and 1")

    total_rows = sum(group.rows for group in groups.values())
    available_labels = Counter()
    for group in groups.values():
        available_labels.update(group.labels)
    if not available_labels[0] or not available_labels[1]:
        raise ValueError("training assignments must contain both human and AI labels")

    target_rows = min(target_rows, total_rows)
    target_human_rows = min(round(target_rows * human_fraction), available_labels[0])
    if target_human_rows < 1:
        raise ValueError("not enough human rows to build the requested pilot")

    # Randomization only resolves exact ties; all meaningful choices are made
    # by the deterministic cost function.
    rng = random.Random(seed)
    candidates = [group for group in groups.values() if group.labels.get(0, 0) > 0]
    rng.shuffle(candidates)
    selected: Set[str] = set()
    current_human_rows = 0
    covered: Set[Tuple[str, str]] = set()
    all_coverage = set().union(*(_coverage(group) for group in candidates))

    while candidates and current_human_rows < target_human_rows:
        missing_coverage = all_coverage - covered
        scored = [
            (
                _selection_cost(
                    group,
                    current_human_rows,
                    target_human_rows,
                    missing_coverage,
                ),
                group.group_id,
                group,
            )
            for group in candidates
        ]
        _, _, chosen = min(scored, key=lambda item: (item[0], item[1]))
        selected.add(chosen.group_id)
        current_human_rows += chosen.labels.get(0, 0)
        covered.update(_coverage(chosen))
        candidates = [group for group in candidates if group.group_id != chosen.group_id]

    if not selected:
        raise ValueError("could not select any groups")

    selection_summary = {
        "target_rows": target_rows,
        "target_human_fraction": human_fraction,
        "target_human_rows": target_human_rows,
        "selected_groups": len(selected),
        "selected_human_capacity": current_human_rows,
        "available_rows": total_rows,
        "available_labels": {str(label): available_labels.get(label, 0) for label in (0, 1)},
        "coverage": {
            category: sorted(value for kind, value in covered if kind == category)
            for category in ("model", "domain", "attack")
        },
        "seed": seed,
        "selection_unit": "training groups for partition safety; records sampled within groups",
    }
    return selected, selection_summary


def _stable_rank(row: Mapping[str, str], seed: int) -> str:
    """Return a deterministic rank used for reproducible row sampling."""

    value = f"{seed}:{row['record_id']}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _row_coverage(row: Mapping[str, str]) -> Set[Tuple[str, str]]:
    return {
        ("model", row.get("model") or "unknown"),
        ("domain", row.get("domain") or "unknown"),
        ("attack", row.get("attack") or "none"),
    }


def _sample_rows(
    rows: Sequence[Mapping[str, str]],
    limit: int,
    seed: int,
) -> List[Mapping[str, str]]:
    """Sample rows deterministically while covering metadata categories."""

    if limit < 1:
        return []
    if len(rows) <= limit:
        return list(rows)

    ranked = sorted(rows, key=lambda row: _stable_rank(row, seed))
    remaining = list(ranked)
    available_coverage = set().union(*(_row_coverage(row) for row in remaining))
    covered: Set[Tuple[str, str]] = set()
    chosen: List[Mapping[str, str]] = []

    # Cover each model/domain/attack category before filling the remainder.
    # This is a coverage heuristic, not a claim that all combinations are
    # equally represented.
    while remaining and len(chosen) < limit and covered != available_coverage:
        candidate_index, candidate = min(
            enumerate(remaining),
            key=lambda item: (
                -len(_row_coverage(item[1]) - covered),
                _stable_rank(item[1], seed),
            ),
        )
        chosen.append(candidate)
        covered.update(_row_coverage(candidate))
        remaining.pop(candidate_index)

    chosen.extend(remaining[: max(0, limit - len(chosen))])
    return chosen


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_pilot(
    rows: Iterable[Mapping[str, str]],
    selected_groups: Set[str],
    output_dir: Path,
    selection_summary: Mapping[str, object],
    human_fraction: float = DEFAULT_HUMAN_FRACTION,
    seed: int = 42,
) -> Dict[str, object]:
    """Sample and write balanced rows from the selected training groups."""

    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "pilot_assignments.csv.gz"
    record_ids_path = output_dir / "pilot_record_ids.txt"
    summary = dict(selection_summary)
    rows_by_group: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("group_id")
        if group_id in selected_groups:
            rows_by_group[group_id].append(row)

    if not rows_by_group:
        raise ValueError("selected groups produced no rows")

    human_candidates = [
        row for group_rows in rows_by_group.values() for row in group_rows if row["label"] == "0"
    ]
    ai_candidates = [
        row for group_rows in rows_by_group.values() for row in group_rows if row["label"] == "1"
    ]
    target_human_rows = int(summary["target_human_rows"])
    selected_human = _sample_rows(human_candidates, target_human_rows, seed)
    target_ai_rows = round(len(selected_human) * (1.0 - human_fraction) / human_fraction)
    selected_ai = _sample_rows(ai_candidates, target_ai_rows, seed + 1)

    if len(selected_human) != target_human_rows:
        raise ValueError(
            f"could not select {target_human_rows} human rows; found {len(selected_human)}"
        )
    if len(selected_ai) != target_ai_rows:
        raise ValueError(
            f"could not select {target_ai_rows} AI rows from selected training groups; "
            f"found {len(selected_ai)}"
        )

    selected_rows = sorted(
        [*selected_human, *selected_ai],
        key=lambda row: _stable_rank(row, seed + 2),
    )
    counts = {
        "models": Counter(),
        "domains": Counter(),
        "attacks": Counter(),
        "labels": Counter(),
    }
    groups_seen = {row["group_id"] for row in selected_rows}
    fieldnames = list(selected_rows[0].keys())

    with gzip.open(assignment_path, "wt", encoding="utf-8", newline="") as assignment_handle, record_ids_path.open(
        "w", encoding="utf-8"
    ) as ids_handle:
        writer = csv.DictWriter(assignment_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)
            ids_handle.write(f"{row['record_id']}\n")
            counts["labels"][row["label"]] += 1
            for field, key in (("models", "model"), ("domains", "domain"), ("attacks", "attack")):
                counts[field][row.get(key) or "unknown"] += 1

    summary.update(
        {
            "target_ai_rows": target_ai_rows,
            "rows_written": len(selected_rows),
            "groups_written": len(groups_seen),
            "selected_groups_with_rows": len(groups_seen),
            "outputs": {
                "assignments": assignment_path.name,
                "record_ids": record_ids_path.name,
                "summary": "pilot_summary.json",
            },
            "distributions": {name: dict(counter) for name, counter in counts.items()},
        }
    )
    _write_json(output_dir / "pilot_summary.json", summary)
    return summary


def build_pilot(
    manifest_dir: Path,
    output_dir: Path,
    target_rows: int = DEFAULT_TARGET_ROWS,
    human_fraction: float = DEFAULT_HUMAN_FRACTION,
    seed: int = 42,
) -> Dict[str, object]:
    """Build a group-safe pilot from local train assignment shards."""

    groups = summarize_groups(read_assignment_rows(manifest_dir, split="train"))
    selected_groups, selection_summary = select_groups(
        groups,
        target_rows=target_rows,
        human_fraction=human_fraction,
        seed=seed,
    )
    selection_summary["partition_overlap_checks"] = verify_group_partition_safety(
        manifest_dir,
        selected_groups,
    )
    return write_pilot(
        read_assignment_rows(manifest_dir, split="train"),
        selected_groups,
        output_dir,
        selection_summary,
        human_fraction=human_fraction,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--human-fraction", type=float, default=DEFAULT_HUMAN_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_pilot(
        args.manifest_dir,
        args.output_dir,
        target_rows=args.target_rows,
        human_fraction=args.human_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
