"""RAID dataset audit job for Kaggle.

This script intentionally audits the data before any model training begins.
It supports two input modes:

1. An attached CSV mounted under /kaggle/input.
2. The official Hugging Face dataset loaded as a streaming dataset.

Configuration is supplied through environment variables so the same script can
be submitted repeatedly without editing the source:

    RAID_DATA_MODE=auto|csv|huggingface
    RAID_DATA_PATH=/path/to/train.csv
    RAID_HF_DATASET=liamdugan/raid
    RAID_SPLIT=train
    RAID_MAX_ROWS=50000
    RAID_SHUFFLE_BUFFER=10000
    RAID_SEED=42
    RAID_OUTPUT_DIR=/kaggle/working
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


DEFAULT_HF_DATASET = "liamdugan/raid"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_ROWS = 50_000
DEFAULT_SHUFFLE_BUFFER = 10_000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "/kaggle/working" if Path("/kaggle/working").exists() else "artifacts"


def env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def parse_label(row: Mapping[str, Any]) -> int | None:
    """Return the binary label when the row contains enough information.

    RAID training rows can be labeled directly or inferred from the model
    field. Test rows may have neither a label nor a human marker, in which case
    the correct result is None rather than an invented label.
    """
    raw_label = row.get("label")
    if raw_label is not None and str(raw_label).strip() != "":
        normalized = str(raw_label).strip().lower()
        if normalized in {"0", "human", "false"}:
            return 0
        if normalized in {"1", "ai", "generated", "true"}:
            return 1

    model = str(row.get("model") or "").strip().lower()
    if model == "human":
        return 0
    if model:
        return 1
    return None


def text_from_row(row: Mapping[str, Any]) -> str:
    """Extract the text field while supporting the project's known schemas."""
    value = row.get("generation") or row.get("text") or row.get("content") or ""
    return str(value)


def find_attached_csv() -> Path | None:
    """Find a likely RAID CSV in Kaggle's mounted input directory."""
    search_roots = [Path("/kaggle/input"), Path("data")]
    candidates: list[Path] = []

    for root in search_roots:
        if root.exists():
            candidates.extend(root.rglob("*.csv"))

    if not candidates:
        return None

    preferred_names = ("train.csv", "raid_train_sample.csv", "raid.csv")
    for preferred_name in preferred_names:
        for candidate in candidates:
            if candidate.name == preferred_name:
                return candidate
    return sorted(candidates)[0]


def iter_csv_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream CSV rows without loading the complete file into memory."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for row in reader:
            yield row


def iter_huggingface_rows(
    dataset_name: str,
    split: str,
    shuffle_buffer: int,
    seed: int,
) -> Iterator[Mapping[str, Any]]:
    """Stream rows from the official Hugging Face RAID dataset."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The datasets package is required for Hugging Face mode. "
            "Install it in the Kaggle environment or attach a CSV dataset."
        ) from error

    dataset = load_dataset(dataset_name, split=split, streaming=True)

    # Shuffling an IterableDataset uses a bounded buffer. It prevents the audit
    # from seeing only the first domain/model block while avoiding a full
    # materialization of the dataset.
    if shuffle_buffer > 1:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

    yield from dataset


def choose_rows() -> tuple[str, Iterator[Mapping[str, Any]]]:
    """Select the configured data source and return a lazy row iterator."""
    mode = os.getenv("RAID_DATA_MODE", "auto").strip().lower()
    dataset_name = os.getenv("RAID_HF_DATASET", DEFAULT_HF_DATASET)
    split = os.getenv("RAID_SPLIT", DEFAULT_SPLIT)
    shuffle_buffer = env_int("RAID_SHUFFLE_BUFFER", DEFAULT_SHUFFLE_BUFFER)
    seed = env_int("RAID_SEED", DEFAULT_SEED)

    configured_path = os.getenv("RAID_DATA_PATH")
    csv_path = Path(configured_path) if configured_path else find_attached_csv()

    if mode not in {"auto", "csv", "huggingface"}:
        raise ValueError("RAID_DATA_MODE must be auto, csv, or huggingface")

    if mode in {"auto", "csv"} and csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Configured RAID_DATA_PATH does not exist: {csv_path}")
        return f"csv:{csv_path}", iter_csv_rows(csv_path)

    if mode == "csv":
        raise FileNotFoundError(
            "CSV mode was requested, but no CSV was found. Set RAID_DATA_PATH "
            "or attach a Kaggle dataset containing a CSV."
        )

    return (
        f"huggingface:{dataset_name}/{split}",
        iter_huggingface_rows(dataset_name, split, shuffle_buffer, seed),
    )


def runtime_report() -> dict[str, Any]:
    """Collect environment information without requiring PyTorch."""
    report: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "kaggle_environment": Path("/kaggle").exists(),
    }

    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            report["cuda_devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        report["torch_available"] = False
    except Exception as error:
        report["torch_error"] = repr(error)

    return report


def audit_rows(
    rows: Iterable[Mapping[str, Any]],
    max_rows: int,
) -> dict[str, Any]:
    """Collect bounded schema, label, metadata, and text statistics."""
    row_count = 0
    labeled_count = 0
    empty_text_count = 0
    missing_field_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    decoding_counts: Counter[str] = Counter()
    repetition_counts: Counter[str] = Counter()
    source_ids: set[str] = set()
    adversarial_source_ids: set[str] = set()
    record_ids: set[str] = set()
    text_hashes: set[str] = set()
    duplicate_record_ids = 0
    duplicate_texts = 0
    sample_rows: list[dict[str, Any]] = []
    text_lengths: list[int] = []

    required_fields = {
        "id",
        "source_id",
        "adv_source_id",
        "model",
        "decoding",
        "repetition_penalty",
        "attack",
        "domain",
        "generation",
    }

    import hashlib

    observed_fields: set[str] = set()

    for row in rows:
        if row_count >= max_rows:
            break

        row_count += 1
        observed_fields.update(str(key) for key in row.keys())

        for field in required_fields:
            if row.get(field) in (None, ""):
                missing_field_counts[field] += 1

        text = text_from_row(row).strip()
        if not text:
            empty_text_count += 1
        text_lengths.append(len(text))

        label = parse_label(row)
        if label is None:
            label_counts["unknown"] += 1
        else:
            label_counts[str(label)] += 1
            labeled_count += 1

        model_counts[str(row.get("model") or "missing")] += 1
        domain_counts[str(row.get("domain") or "missing")] += 1
        attack_counts[str(row.get("attack") or "missing")] += 1
        decoding_counts[str(row.get("decoding") or "missing")] += 1
        repetition_counts[str(row.get("repetition_penalty") or "missing")] += 1

        record_id = str(row.get("id") or "")
        if record_id:
            if record_id in record_ids:
                duplicate_record_ids += 1
            record_ids.add(record_id)

        source_id = str(row.get("source_id") or "")
        if source_id:
            source_ids.add(source_id)

        adversarial_source_id = str(row.get("adv_source_id") or "")
        if adversarial_source_id:
            adversarial_source_ids.add(adversarial_source_id)

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in text_hashes:
            duplicate_texts += 1
        text_hashes.add(text_hash)

        if len(sample_rows) < 10:
            sample_rows.append(
                {
                    "id": record_id,
                    "label": label,
                    "model": row.get("model"),
                    "domain": row.get("domain"),
                    "attack": row.get("attack"),
                    "decoding": row.get("decoding"),
                    "text_preview": text[:300],
                }
            )

    def length_stats(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 2),
        }

    return {
        "rows_audited": row_count,
        "labeled_rows": labeled_count,
        "observed_fields": sorted(observed_fields),
        "missing_field_counts": dict(sorted(missing_field_counts.items())),
        "label_counts": dict(label_counts),
        "model_counts": dict(model_counts),
        "domain_counts": dict(domain_counts),
        "attack_counts": dict(attack_counts),
        "decoding_counts": dict(decoding_counts),
        "repetition_penalty_counts": dict(repetition_counts),
        "unique_record_ids": len(record_ids),
        "unique_source_ids": len(source_ids),
        "unique_adv_source_ids": len(adversarial_source_ids),
        "duplicate_record_ids": duplicate_record_ids,
        "duplicate_texts": duplicate_texts,
        "empty_text_rows": empty_text_count,
        "text_character_lengths": length_stats(text_lengths),
        "samples": sample_rows,
    }


def main() -> None:
    max_rows = env_int("RAID_MAX_ROWS", DEFAULT_MAX_ROWS)
    output_dir = Path(os.getenv("RAID_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    source, rows = choose_rows()
    report = {
        "audit_version": 1,
        "source": source,
        "configuration": {
            "max_rows": max_rows,
            "split": os.getenv("RAID_SPLIT", DEFAULT_SPLIT),
            "seed": env_int("RAID_SEED", DEFAULT_SEED),
        },
        "runtime": runtime_report(),
        "data": audit_rows(rows, max_rows),
    }

    report_path = output_dir / "raid_audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nAudit report written to: {report_path}")


if __name__ == "__main__":
    main()
