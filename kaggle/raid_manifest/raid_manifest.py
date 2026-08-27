"""Create a reusable, leakage-safe RAID manifest on Kaggle.

The raw RAID text stays in the Hugging Face dataset.  This job stores only the
derived assignment needed by later training jobs:

    record_id, group_id, label, split, and evaluation metadata

The stream is materialized into a temporary SQLite database because the full
dataset is too large to keep as Python dictionaries. The database is deleted
after the compressed manifest and summary are written. Exact text hashes with
contradictory labels are quarantined and reported.

Environment variables:

    RAID_DATA_MODE=auto|csv|huggingface
    RAID_DATA_PATH=/path/to/train.csv       # csv mode
    RAID_HF_DATASET=liamdugan/raid
    RAID_SPLIT=train
    RAID_MAX_ROWS=0                         # 0 means the full split
    RAID_SEED=42
    RAID_SHARD_COUNT=32                     # number of shards per split
    RAID_OUTPUT_DIR=/kaggle/working
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SPLITS = ("train", "validation", "test")
DEFAULT_DATASET = "liamdugan/raid"
DEFAULT_SPLIT = "train"


class UnionFind:
    """Union-find over source and adversarial lineage identifiers."""

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
        if left_root != right_root:
            if left_root < right_root:
                self.parent[right_root] = left_root
            else:
                self.parent[left_root] = right_root


def optional_string(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def required_string(value: object, field: str, row_number: int) -> str:
    result = optional_string(value)
    if result is None:
        raise ValueError(f"row {row_number} is missing required field '{field}'")
    return result


def label_from_model(model: str) -> int:
    normalized = model.strip().lower()
    if not normalized:
        raise ValueError("model cannot be empty")
    return 0 if normalized == "human" else 1


def row_links(record_id: str, source_id: Optional[str], adv_source_id: Optional[str]) -> List[str]:
    links = []
    if source_id:
        links.append(f"source:{source_id}")
    if adv_source_id:
        links.append(f"adv:{adv_source_id}")
    return links or [f"record:{record_id}"]


def iter_csv_rows(path: Path, max_rows: Optional[int]) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        missing = {"id", "model"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            yield row


def iter_huggingface_rows(
    dataset_name: str, split: str, max_rows: Optional[int]
) -> Iterable[Mapping[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Kaggle environment needs the 'datasets' package") from error

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    for index, row in enumerate(dataset):
        if max_rows is not None and index >= max_rows:
            break
        yield row


def choose_rows(
    mode: str,
    data_path: Optional[Path],
    dataset_name: str,
    split: str,
    max_rows: Optional[int],
) -> Iterable[Mapping[str, object]]:
    if mode not in {"auto", "csv", "huggingface"}:
        raise ValueError("RAID_DATA_MODE must be auto, csv, or huggingface")
    if mode == "auto":
        mode = "csv" if data_path and data_path.exists() else "huggingface"
    if mode == "csv":
        if data_path is None:
            raise ValueError("RAID_DATA_PATH is required in csv mode")
        return iter_csv_rows(data_path, max_rows)
    return iter_huggingface_rows(dataset_name, split, max_rows)


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            text_hash TEXT NOT NULL,
            text_length INTEGER NOT NULL,
            label INTEGER NOT NULL,
            source_id TEXT,
            adv_source_id TEXT,
            model TEXT NOT NULL,
            decoding TEXT,
            repetition_penalty TEXT,
            attack TEXT NOT NULL,
            domain TEXT NOT NULL,
            group_id TEXT,
            split TEXT
        );
        CREATE TABLE ids (record_id TEXT PRIMARY KEY);
        CREATE TABLE text_index (
            text_hash TEXT PRIMARY KEY,
            label INTEGER NOT NULL,
            first_record_id TEXT NOT NULL,
            first_links TEXT NOT NULL,
            conflicting INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX records_source_idx ON records(source_id);
        CREATE INDEX records_adv_source_idx ON records(adv_source_id);
        CREATE INDEX records_group_idx ON records(group_id);
        """
    )


def ingest_rows(
    connection: sqlite3.Connection,
    rows: Iterable[Mapping[str, object]],
    union_find: UnionFind,
) -> Dict[str, int]:
    stats = {
        "rows_seen": 0,
        "rows_kept": 0,
        "empty_rows_removed": 0,
        "duplicate_texts_removed": 0,
        "conflicting_texts_removed": 0,
        "conflicting_rows_removed": 0,
    }
    connection.execute("BEGIN")

    for row_number, row in enumerate(rows, start=2):
        stats["rows_seen"] += 1
        record_id = required_string(row.get("id"), "id", row_number)
        model = required_string(row.get("model"), "model", row_number)
        raw_text = row.get("generation")
        if raw_text is None:
            raw_text = row.get("text")
        text = "" if raw_text is None else str(raw_text)
        if not text.strip():
            stats["empty_rows_removed"] += 1
            continue

        source_id = optional_string(row.get("source_id"))
        adv_source_id = optional_string(row.get("adv_source_id"))
        links = row_links(record_id, source_id, adv_source_id)
        for link in links:
            union_find.add(link)
        for link in links[1:]:
            union_find.union(links[0], link)

        try:
            connection.execute("INSERT INTO ids(record_id) VALUES (?)", (record_id,))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"duplicate record id encountered: {record_id}") from error

        label = label_from_model(model)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        previous = connection.execute(
            "SELECT label, first_record_id, first_links, conflicting FROM text_index WHERE text_hash = ?",
            (text_hash,),
        ).fetchone()
        if previous is not None:
            # Preserve lineage even when the duplicate row is dropped.
            for first_link in json.loads(previous[2]):
                union_find.union(links[0], first_link)

            if int(previous[3]) == 1:
                stats["conflicting_rows_removed"] += 1
                continue

            if int(previous[0]) != label:
                connection.execute(
                    "UPDATE text_index SET conflicting = 1 WHERE text_hash = ?",
                    (text_hash,),
                )
                connection.execute("DELETE FROM records WHERE record_id = ?", (previous[1],))
                stats["rows_kept"] -= 1
                stats["conflicting_texts_removed"] += 1
                stats["conflicting_rows_removed"] += 2
                continue

            stats["duplicate_texts_removed"] += 1
            continue

        connection.execute(
            "INSERT INTO text_index(text_hash, label, first_record_id, first_links, conflicting) VALUES (?, ?, ?, ?, 0)",
            (text_hash, label, record_id, json.dumps(links)),
        )
        connection.execute(
            """
            INSERT INTO records(
                record_id, text_hash, text_length, label, source_id, adv_source_id,
                model, decoding, repetition_penalty, attack, domain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                text_hash,
                len(text),
                label,
                source_id,
                adv_source_id,
                model,
                optional_string(row.get("decoding")),
                optional_string(row.get("repetition_penalty")),
                optional_string(row.get("attack")) or "none",
                optional_string(row.get("domain")) or "unknown",
            ),
        )
        stats["rows_kept"] += 1

        if stats["rows_seen"] % 10000 == 0:
            connection.commit()
            connection.execute("BEGIN")
            print(f"Ingested {stats['rows_seen']:,} rows; kept {stats['rows_kept']:,}")

    connection.commit()
    return stats


def assign_group_ids(connection: sqlite3.Connection, union_find: UnionFind) -> int:
    group_names: Dict[str, str] = {}
    updates = []

    for record_id, source_id, adv_source_id in connection.execute(
        "SELECT record_id, source_id, adv_source_id FROM records"
    ):
        links = row_links(record_id, source_id, adv_source_id)
        root = union_find.find(links[0])
        if root not in group_names:
            group_names[root] = "group-" + hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]
        updates.append((group_names[root], record_id))
        if len(updates) >= 10000:
            connection.executemany("UPDATE records SET group_id = ? WHERE record_id = ?", updates)
            connection.commit()
            updates.clear()

    if updates:
        connection.executemany("UPDATE records SET group_id = ? WHERE record_id = ?", updates)
        connection.commit()
    return len(group_names)


def load_group_stats(connection: sqlite3.Connection) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, Dict[str, object]] = {}
    for group_id, size, human_count, ai_count in connection.execute(
        """
        SELECT group_id, COUNT(*),
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END)
        FROM records GROUP BY group_id
        """
    ):
        groups[str(group_id)] = {
            "size": int(size),
            "labels": {0: int(human_count or 0), 1: int(ai_count or 0)},
        }
    return groups


def assign_splits(
    connection: sqlite3.Connection,
    groups: Mapping[str, Mapping[str, object]],
    seed: int,
) -> None:
    if len(groups) < len(SPLITS):
        raise ValueError(f"need at least {len(SPLITS)} groups; found {len(groups)}")

    total_rows = sum(int(group["size"]) for group in groups.values())
    total_labels = Counter()
    for group in groups.values():
        total_labels.update(group["labels"])

    ratios = {"train": 0.80, "validation": 0.10, "test": 0.10}
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: -int(item[1]["size"]))
    remaining = list(items)
    assignments: Dict[str, str] = {}
    row_counts = {split: 0 for split in SPLITS}
    label_counts = {split: Counter() for split in SPLITS}

    def seed_cost(item: Tuple[str, Mapping[str, object]], split: str) -> Tuple[float, str]:
        group_id, group = item
        target_rows = total_rows * ratios[split]
        target_labels = {label: total_labels[label] * ratios[split] for label in (0, 1)}
        size_error = abs(int(group["size"]) - target_rows) / max(target_rows, 1.0)
        label_error = sum(
            abs(int(group["labels"].get(label, 0)) - target_labels[label])
            / max(target_labels[label], 1.0)
            for label in (0, 1)
        )
        return size_error + label_error, group_id

    for split in SPLITS:
        selected_id, selected = min(remaining, key=lambda item: seed_cost(item, split))
        remaining = [item for item in remaining if item[0] != selected_id]
        assignments[selected_id] = split
        row_counts[split] += int(selected["size"])
        label_counts[split].update(selected["labels"])

    for group_id, group in remaining:
        choices = []
        group_labels = group["labels"]
        for split in SPLITS:
            target_rows = total_rows * ratios[split]
            target_labels = {label: total_labels[label] * ratios[split] for label in (0, 1)}
            new_rows = row_counts[split] + int(group["size"])
            cost = abs(new_rows - target_rows) / max(target_rows, 1.0)
            cost += sum(
                2.0
                * abs(label_counts[split].get(label, 0) + int(group_labels.get(label, 0)) - target_labels[label])
                / max(target_labels[label], 1.0)
                for label in (0, 1)
            )
            missing = [label for label in (0, 1) if label_counts[split].get(label, 0) == 0]
            cost -= 3.0 * sum(1 for label in missing if group_labels.get(label, 0) > 0)
            choices.append((cost, split))

        _, selected_split = min(choices)
        assignments[group_id] = selected_split
        row_counts[selected_split] += int(group["size"])
        label_counts[selected_split].update(group_labels)

    connection.executemany(
        "UPDATE records SET split = ? WHERE group_id = ?",
        [(split, group_id) for group_id, split in assignments.items()],
    )
    connection.commit()


def validate(connection: sqlite3.Connection) -> None:
    for split in SPLITS:
        labels = {
            int(label)
            for (label,) in connection.execute(
                "SELECT DISTINCT label FROM records WHERE split = ?", (split,)
            )
        }
        if labels != {0, 1}:
            raise ValueError(f"split '{split}' does not contain both labels: {labels}")

    checks = (
        ("group_id", "group leakage"),
        ("source_id", "source_id leakage"),
        ("adv_source_id", "adv_source_id leakage"),
        ("text_hash", "text leakage"),
    )
    for field, description in checks:
        query = (
            f"SELECT {field} FROM records WHERE {field} IS NOT NULL "
            f"GROUP BY {field} HAVING COUNT(DISTINCT split) > 1 LIMIT 1"
        )
        violation = connection.execute(query).fetchone()
        if violation is not None:
            raise ValueError(f"{description} detected for {violation[0]}")


def distribution(connection: sqlite3.Connection) -> Dict[str, object]:
    result: Dict[str, object] = {"splits": {}}
    for split in SPLITS:
        split_data: Dict[str, object] = {"rows": 0, "labels": {}, "models": {}, "domains": {}, "attacks": {}}
        split_data["rows"] = connection.execute(
            "SELECT COUNT(*) FROM records WHERE split = ?", (split,)
        ).fetchone()[0]
        for output_key, column in (("labels", "label"), ("models", "model"), ("domains", "domain"), ("attacks", "attack")):
            split_data[output_key] = {
                str(value): count
                for value, count in connection.execute(
                    f"SELECT {column}, COUNT(*) FROM records WHERE split = ? GROUP BY {column}",
                    (split,),
                )
            }
        result["splits"][split] = split_data
    result["groups"] = connection.execute("SELECT COUNT(DISTINCT group_id) FROM records").fetchone()[0]
    return result


def write_outputs(
    connection: sqlite3.Connection,
    output_dir: Path,
    config: Mapping[str, object],
    ingest_stats: Mapping[str, int],
    shard_count: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")

    columns = [
        "record_id", "group_id", "label", "split", "source_id", "adv_source_id",
        "model", "domain", "attack", "decoding", "repetition_penalty", "text_length", "text_hash",
    ]

    # A single multi-million-row output file was not reliably downloadable from
    # Kaggle. Keep each compressed file bounded and provide an index for later
    # training jobs. Shards are deterministic from the record text hash.
    shard_writers: Dict[Tuple[str, int], csv.writer] = {}
    shard_counts: Counter = Counter()
    shard_metadata: List[Dict[str, object]] = []

    with ExitStack() as stack:
        for split in SPLITS:
            query = (
                "SELECT record_id, group_id, label, split, source_id, adv_source_id, "
                "model, domain, attack, decoding, repetition_penalty, text_length, text_hash "
                "FROM records WHERE split = ? ORDER BY record_id"
            )
            for row in connection.execute(query, (split,)):
                text_hash = str(row[-1])
                shard = int(text_hash[:8], 16) % shard_count
                key = (split, shard)
                if key not in shard_writers:
                    filename = f"record_assignments_{split}_{shard:02d}.csv.gz"
                    handle = stack.enter_context(
                        gzip.open(output_dir / filename, "wt", encoding="utf-8", newline="")
                    )
                    writer = csv.writer(handle)
                    writer.writerow(columns)
                    shard_writers[key] = writer
                    shard_metadata.append({"split": split, "shard": shard, "file": filename})
                shard_writers[key].writerow(row)
                shard_counts[key] += 1

    for item in shard_metadata:
        item["rows"] = shard_counts[(str(item["split"]), int(item["shard"]))]

    index = {
        "manifest_version": 1,
        "shard_count_per_split": shard_count,
        "columns": columns,
        "shards": sorted(shard_metadata, key=lambda item: (str(item["split"]), int(item["shard"]))),
    }
    index_path = output_dir / "manifest_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, ensure_ascii=False)

    report: Dict[str, object] = {
        "manifest_version": 1,
        "configuration": dict(config),
        "preprocessing": dict(ingest_stats),
        "data": distribution(connection),
        "outputs": {
            "assignment_index": index_path.name,
            "assignment_shards": len(shard_metadata),
        },
    }
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (output_dir / "manifest_config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=2, ensure_ascii=False)
    return report


def build_manifest(
    mode: str,
    data_path: Optional[Path],
    dataset_name: str,
    split: str,
    max_rows: Optional[int],
    seed: int,
    shard_count: int,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="raid_manifest_") as temporary_dir:
        database_path = Path(temporary_dir) / "manifest.sqlite"
        connection = sqlite3.connect(database_path)
        create_schema(connection)
        union_find = UnionFind()
        rows = choose_rows(mode, data_path, dataset_name, split, max_rows)
        ingest_stats = ingest_rows(connection, rows, union_find)
        group_count = assign_group_ids(connection, union_find)
        groups = load_group_stats(connection)
        assign_splits(connection, groups, seed)
        validate(connection)
        config = {
            "source": f"huggingface:{dataset_name}/{split}" if mode != "csv" else str(data_path),
            "dataset": dataset_name,
            "split": split,
            "max_rows": max_rows,
            "seed": seed,
            "shard_count_per_split": shard_count,
            "group_count": group_count,
            "grouping": "connected source_id and adv_source_id lineage",
            "label_rule": "model == human -> 0; otherwise -> 1",
        }
        report = write_outputs(connection, output_dir, config, ingest_stats, shard_count)
        connection.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "csv", "huggingface"), default=os.getenv("RAID_DATA_MODE", "auto"))
    parser.add_argument("--input", type=Path, default=Path(os.getenv("RAID_DATA_PATH", "")) if os.getenv("RAID_DATA_PATH") else None)
    parser.add_argument("--dataset", default=os.getenv("RAID_HF_DATASET", DEFAULT_DATASET))
    parser.add_argument("--split", default=os.getenv("RAID_SPLIT", DEFAULT_SPLIT))
    parser.add_argument("--max-rows", type=int, default=int(os.getenv("RAID_MAX_ROWS", "0")) or None)
    parser.add_argument("--seed", type=int, default=int(os.getenv("RAID_SEED", "42")))
    parser.add_argument("--shard-count", type=int, default=int(os.getenv("RAID_SHARD_COUNT", "32")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("RAID_OUTPUT_DIR", "/kaggle/working")))
    args = parser.parse_args()

    report = build_manifest(
        mode=args.mode,
        data_path=args.input,
        dataset_name=args.dataset,
        split=args.split,
        max_rows=args.max_rows,
        seed=args.seed,
        shard_count=args.shard_count,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
