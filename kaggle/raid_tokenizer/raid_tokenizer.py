"""Run the first real RoBERTa tokenizer audit on Kaggle.

The local implementation lives in ``src/training/tokenizer_validation.py``.
This Kaggle entrypoint is intentionally self-contained because Kaggle executes
the kernel entrypoint rather than the local repository package.  It writes both
``tokenizer_validation_report.json`` and ``tokenizer_validation.log`` to the
working directory so the run can be inspected after download.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


MODEL_NAME = "FacebookAI/roberta-base"
MAX_LENGTH = 512
PROGRESS_EVERY = 1000
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working")
REPORT_PATH = OUTPUT_ROOT / "tokenizer_validation_report.json"
LOG_PATH = OUTPUT_ROOT / "tokenizer_validation.log"
LOGGER = logging.getLogger("raid_tokenizer")


def configure_logging() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def find_joined_file() -> Path:
    accepted_names = {
        "raid_pilot_joined.jsonl.gz",
        "raid_pilot_joined.jsonl",
        ".raid_pilot_joined.jsonl.gz.tmp",
    }
    matches = sorted(
        path
        for path in INPUT_ROOT.rglob("*")
        if path.is_file() and path.name in accepted_names
    )
    if not matches:
        raise FileNotFoundError(
            f"could not find the joined pilot JSONL below {INPUT_ROOT}"
        )
    if len(matches) > 1:
        raise ValueError(f"found multiple joined pilot files: {matches}")
    return matches[0]


def rows_from_jsonl(path: Path):
    with path.open("rb") as probe:
        is_gzip = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    LOGGER.info("reading joined pilot: compression=%s", "gzip" if is_gzip else "plain JSONL")
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def one_dimensional(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenizer output row")
        return list(value[0])
    return list(value or [])


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def length_summary(values):
    if not values:
        raise ValueError("cannot summarize empty lengths")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def validate(rows, tokenizer):
    all_lengths = []
    lengths_by_label = defaultdict(list)
    labels = Counter()
    metadata = {field: Counter() for field in ("model", "domain", "attack")}
    examples = []
    seen_ids = set()
    truncated = 0
    empty = 0

    LOGGER.info("validation started: model=%s max_length=%d", MODEL_NAME, MAX_LENGTH)
    for row_number, row in enumerate(rows, start=1):
        required = {"record_id", "text", "label"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {row_number} missing columns: {sorted(missing)}")
        record_id = str(row["record_id"])
        if not record_id or record_id in seen_ids:
            raise ValueError(f"empty or duplicate record_id at row {row_number}: {record_id!r}")
        seen_ids.add(record_id)
        if row["label"] not in (0, 1, "0", "1"):
            raise ValueError(f"invalid label at row {row_number}: {row['label']!r}")
        label = int(row["label"])
        text = "" if row["text"] is None else str(row["text"])
        empty += int(not text.strip())

        full = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)
        full_ids = one_dimensional(full["input_ids"])
        if not full_ids:
            raise ValueError(f"tokenizer returned no IDs at row {row_number}")
        original_length = len(full_ids)

        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        input_ids = one_dimensional(encoded.get("input_ids"))
        attention_mask = one_dimensional(encoded.get("attention_mask"))
        special_mask = one_dimensional(encoded.get("special_tokens_mask"))
        if len(input_ids) != MAX_LENGTH:
            raise ValueError(f"row {row_number} input length is {len(input_ids)}, expected {MAX_LENGTH}")
        if len(attention_mask) != MAX_LENGTH:
            raise ValueError(f"row {row_number} attention mask has wrong length")
        if special_mask and len(special_mask) != MAX_LENGTH:
            raise ValueError(f"row {row_number} special-token mask has wrong length")

        was_truncated = original_length > MAX_LENGTH
        truncated += int(was_truncated)
        labels[str(label)] += 1
        all_lengths.append(original_length)
        lengths_by_label[str(label)].append(original_length)
        for field in metadata:
            value = row.get(field)
            metadata[field][str(value) if value not in (None, "") else "unknown"] += 1
        if len(examples) < 3:
            examples.append({
                "record_id": record_id,
                "label": label,
                "text_preview": text[:240],
                "tokens": tokenizer.convert_ids_to_tokens(full_ids),
                "original_token_length": original_length,
                "truncated": was_truncated,
            })
        if labels.total() % PROGRESS_EVERY == 0:
            LOGGER.info(
                "progress: rows=%d human=%d ai=%d truncated=%d empty=%d",
                labels.total(), labels.get("0", 0), labels.get("1", 0), truncated, empty
            )

    if not all_lengths:
        raise ValueError("joined pilot is empty")
    LOGGER.info(
        "validation completed: rows=%d human=%d ai=%d truncated=%d (%.2f%%) empty=%d",
        len(all_lengths), labels.get("0", 0), labels.get("1", 0), truncated,
        100 * truncated / len(all_lengths), empty
    )
    return {
        "model": MODEL_NAME,
        "tokenizer_class": tokenizer.__class__.__name__,
        "max_length": MAX_LENGTH,
        "rows_seen": len(all_lengths),
        "unique_record_ids": len(seen_ids),
        "labels": dict(sorted(labels.items())),
        "empty_text_rows": empty,
        "truncated_rows": truncated,
        "truncation_rate": round(truncated / len(all_lengths), 6),
        "lengths": length_summary(all_lengths),
        "lengths_by_label": {
            label: length_summary(values) for label, values in sorted(lengths_by_label.items())
        },
        "metadata_counts": {
            field: dict(sorted(counts.items())) for field, counts in sorted(metadata.items())
        },
        "examples": examples,
    }


def main():
    configure_logging()
    try:
        from transformers import AutoTokenizer

        input_path = find_joined_file()
        LOGGER.info("input found: %s", input_path)
        LOGGER.info("loading tokenizer: %s", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        LOGGER.info("tokenizer loaded: class=%s", tokenizer.__class__.__name__)
        report = validate(rows_from_jsonl(input_path), tokenizer)
        report["input_file"] = str(input_path)
        with REPORT_PATH.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        LOGGER.info("report written: %s", REPORT_PATH)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception:
        LOGGER.exception("tokenizer validation failed")
        raise


if __name__ == "__main__":
    main()
