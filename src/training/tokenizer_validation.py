"""Validate a Hugging Face tokenizer against the joined RAID pilot.

This stage deliberately does not train a model.  It checks that every joined
record can be converted into a fixed-length model input and reports sequence
lengths, truncation, labels, and representative tokenizations.

The module's data and validation helpers use only the Python standard library.
The ``transformers`` dependency is imported only by ``main`` so the helpers can
be unit-tested in the lightweight local environment.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


REQUIRED_COLUMNS = {"record_id", "text", "label"}
DEFAULT_MAX_LENGTH = 512
DEFAULT_MODEL = "FacebookAI/roberta-base"
LOGGER = logging.getLogger(__name__)


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    """Stream records from a JSONL or JSONL.GZ file."""

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield row


def _as_list(value: Any) -> List[Any]:
    """Convert a tokenizer output field to a one-example Python list."""

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected tokenizer output for one example")
        return list(value[0])
    if isinstance(value, list):
        return value
    return list(value)


def _percentile(values: Sequence[int], percentile: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""

    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_lengths(lengths: Sequence[int]) -> Dict[str, float]:
    """Summarize token lengths using stable, JSON-friendly scalar values."""

    if not lengths:
        raise ValueError("cannot summarize an empty length sequence")
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.fmean(lengths), 3),
        "median": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
    }


def _text(row: Mapping[str, Any]) -> str:
    value = row.get("text")
    return "" if value is None else str(value)


def _label(row: Mapping[str, Any], row_number: int) -> int:
    value = row.get("label")
    if value not in (0, 1, "0", "1"):
        raise ValueError(f"row {row_number} has invalid label {value!r}")
    return int(value)


def _metadata_counts(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        counts[str(value) if value not in (None, "") else "unknown"] += 1
    return dict(sorted(counts.items()))


def validate_tokenizer(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    model_name: str = DEFAULT_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    example_limit: int = 3,
    progress_every: int = 1000,
) -> Dict[str, Any]:
    """Validate tokenizer outputs and return a JSON-serializable report.

    ``tokenizer`` must have the standard Hugging Face call interface.  Each
    record is tokenized twice: once without truncation to measure its original
    length, and once with padding/truncation to validate the model input shape.
    """

    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if example_limit < 1:
        raise ValueError("example_limit must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    LOGGER.info(
        "tokenizer validation started: model=%s max_length=%d examples=%d",
        model_name,
        max_length,
        example_limit,
    )

    all_lengths: List[int] = []
    lengths_by_label: MutableMapping[str, List[int]] = defaultdict(list)
    labels: Counter[str] = Counter()
    metadata: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    examples: List[Dict[str, Any]] = []
    truncated_rows = 0
    empty_text_rows = 0
    rows_seen = 0

    for row_number, row in enumerate(rows, start=1):
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise ValueError(f"row {row_number} is missing columns: {sorted(missing)}")
        record_id = str(row["record_id"])
        if not record_id:
            raise ValueError(f"row {row_number} has an empty record_id")
        label = _label(row, row_number)
        text = _text(row)
        if not text.strip():
            empty_text_rows += 1

        untruncated = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        full_ids = _as_list(untruncated.get("input_ids"))
        if not full_ids:
            raise ValueError(f"tokenizer returned no input_ids for row {row_number}")
        original_length = len(full_ids)

        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        input_ids = _as_list(encoded.get("input_ids"))
        attention_mask = _as_list(encoded.get("attention_mask"))
        special_tokens_mask = _as_list(encoded.get("special_tokens_mask"))
        if len(input_ids) != max_length:
            raise ValueError(
                f"row {row_number} produced {len(input_ids)} input IDs; expected {max_length}"
            )
        if len(attention_mask) != max_length:
            raise ValueError(f"row {row_number} has an invalid attention mask length")
        if special_tokens_mask and len(special_tokens_mask) != max_length:
            raise ValueError(f"row {row_number} has an invalid special-token mask length")

        was_truncated = original_length > max_length
        truncated_rows += int(was_truncated)
        rows_seen += 1
        label_key = str(label)
        labels[label_key] += 1
        all_lengths.append(original_length)
        lengths_by_label[label_key].append(original_length)
        for field in ("model", "domain", "attack"):
            value = row.get(field)
            metadata[field][str(value) if value not in (None, "") else "unknown"] += 1

        if len(examples) < example_limit:
            examples.append(
                {
                    "record_id": record_id,
                    "label": label,
                    "text_preview": text[:240],
                    "tokens": tokenizer.convert_ids_to_tokens(full_ids),
                    "original_token_length": original_length,
                    "truncated": was_truncated,
                }
            )

        if rows_seen % progress_every == 0:
            LOGGER.info(
                "progress: rows=%d human=%d ai=%d truncated=%d empty=%d",
                rows_seen,
                labels.get("0", 0),
                labels.get("1", 0),
                truncated_rows,
                empty_text_rows,
            )

    if not rows_seen:
        raise ValueError("no rows were available for tokenizer validation")

    report = {
        "model": model_name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "max_length": max_length,
        "rows_seen": rows_seen,
        "labels": dict(sorted(labels.items())),
        "empty_text_rows": empty_text_rows,
        "truncated_rows": truncated_rows,
        "truncation_rate": round(truncated_rows / rows_seen, 6),
        "lengths": summarize_lengths(all_lengths),
        "lengths_by_label": {
            label: summarize_lengths(values)
            for label, values in sorted(lengths_by_label.items())
        },
        "metadata_counts": {
            field: dict(sorted(counts.items()))
            for field, counts in sorted(metadata.items())
        },
        "examples": examples,
    }
    LOGGER.info(
        "tokenizer validation completed: rows=%d human=%d ai=%d truncated=%d (%.2f%%) empty=%d",
        rows_seen,
        labels.get("0", 0),
        labels.get("1", 0),
        truncated_rows,
        100 * truncated_rows / rows_seen,
        empty_text_rows,
    )
    return report


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Write a tokenizer report as formatted JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    LOGGER.info("loading tokenizer: model=%s", args.model)

    try:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            LOGGER.exception("transformers import failed")
            raise RuntimeError(
                "Tokenizer validation needs transformers; install it locally or run the Kaggle job"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        LOGGER.info("tokenizer loaded: class=%s", tokenizer.__class__.__name__)
        report = validate_tokenizer(
            read_jsonl(args.input),
            tokenizer,
            model_name=args.model,
            max_length=args.max_length,
            example_limit=args.examples,
            progress_every=args.progress_every,
        )
        write_report(report, args.output)
        LOGGER.info("report written: %s", args.output)
    except Exception:
        LOGGER.exception("tokenizer validation failed")
        raise
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
