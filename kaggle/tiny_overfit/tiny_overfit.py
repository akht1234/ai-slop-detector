"""Run the fixed-subset Transformer pipeline overfit test on Kaggle.

This is a wiring test, not a quality benchmark. It uses 64 records (32 human
and 32 AI), trains on those same records, and checks that the model can drive
training loss down and accuracy up. A small randomly initialized RoBERTa-shaped
configuration keeps the test practical on Kaggle's older P100/CPU environment;
the real pretrained baseline model is recorded separately. The complete log
and checkpoint are saved as outputs.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
from pathlib import Path

MODEL_NAME = "roberta-random-tiny-config"
BASELINE_MODEL_NAME = "FacebookAI/roberta-base"
# The tokenizer audit uses 512. The tiny wiring test uses 128 so its CPU
# fallback remains practical when Kaggle assigns an older unsupported GPU.
MAX_LENGTH = 128
ROWS_PER_LABEL = 32
BATCH_SIZE = 8
EPOCHS = 20
# This scratch diagnostic needs a larger rate than the eventual pretrained
# baseline; 5e-5 is appropriate for fine-tuning, not random initialization.
LEARNING_RATE = 1e-3
SEED = 42
INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working")
LOG_PATH = OUTPUT_ROOT / "tiny_overfit.log"
REPORT_PATH = OUTPUT_ROOT / "tiny_overfit_report.json"
CHECKPOINT_PATH = OUTPUT_ROOT / "tiny_overfit_checkpoint"
LOGGER = logging.getLogger("tiny_overfit")


def configure_logging() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")],
        force=True,
    )


def find_joined_file() -> Path:
    accepted_names = {
        "raid_pilot_joined.jsonl.gz",
        "raid_pilot_joined.jsonl",
        ".raid_pilot_joined.jsonl.gz.tmp",
    }
    matches = sorted(
        path for path in INPUT_ROOT.rglob("*") if path.is_file() and path.name in accepted_names
    )
    if not matches:
        raise FileNotFoundError("could not find joined pilot JSONL below /kaggle/input")
    if len(matches) > 1:
        raise ValueError(f"found multiple joined pilot files: {matches}")
    return matches[0]


def read_rows(path: Path):
    with path.open("rb") as probe:
        is_gzip = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    LOGGER.info("reading input: %s compression=%s", path, "gzip" if is_gzip else "plain JSONL")
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} is not an object")
            yield row


def select_fixed_subset(rows):
    selected = {0: [], 1: []}
    for row in rows:
        label = int(row["label"])
        if label in selected and len(selected[label]) < ROWS_PER_LABEL:
            selected[label].append(row)
        if all(len(values) == ROWS_PER_LABEL for values in selected.values()):
            break
    if any(len(values) != ROWS_PER_LABEL for values in selected.values()):
        raise ValueError(f"could not select {ROWS_PER_LABEL} records per label")
    result = selected[0] + selected[1]
    random.Random(SEED).shuffle(result)
    LOGGER.info(
        "fixed subset selected: rows=%d human=%d ai=%d seed=%d",
        len(result),
        sum(int(row["label"]) == 0 for row in result),
        sum(int(row["label"]) == 1 for row in result),
        SEED,
    )
    return result


def train(rows):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig, AutoTokenizer, RobertaForSequenceClassification

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cuda_available = torch.cuda.is_available()
    cuda_capability = None
    device_reason = "CUDA unavailable"
    if cuda_available:
        major, minor = torch.cuda.get_device_capability()
        cuda_capability = f"sm_{major}{minor}"
        if major >= 7:
            device = torch.device("cuda")
            device_reason = "compatible CUDA device"
        else:
            device = torch.device("cpu")
            device_reason = "CUDA device is below PyTorch's supported sm_70 minimum"
            LOGGER.warning(
                "CUDA capability=%s is unsupported by torch=%s; falling back to CPU",
                cuda_capability,
                torch.__version__,
            )
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    LOGGER.info(
        "device=%s reason=%s cuda_available=%s capability=%s torch=%s",
        device, device_reason, cuda_available, cuda_capability, torch.__version__,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_NAME, use_fast=True)
    encodings = tokenizer(
        [str(row["text"]) for row in rows],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)

    class TinyDataset(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, index):
            return {
                "input_ids": encodings["input_ids"][index],
                "attention_mask": encodings["attention_mask"][index],
                "labels": labels[index],
            }

    loader = DataLoader(TinyDataset(), batch_size=BATCH_SIZE, shuffle=False)
    config = AutoConfig.from_pretrained(BASELINE_MODEL_NAME)
    config.num_labels = 2
    config.hidden_size = 64
    config.intermediate_size = 256
    config.num_hidden_layers = 2
    config.num_attention_heads = 2
    config.max_position_embeddings = MAX_LENGTH + 2
    model = RobertaForSequenceClassification(config)
    model.to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    LOGGER.info("parameters: total=%d trainable=%d", total_parameters, trainable_parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        count = 0
        epoch_logits = []
        gradient_norms = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            predictions = outputs.logits.argmax(dim=-1)
            correct += int((predictions == batch["labels"]).sum().item())
            count += int(batch["labels"].shape[0])
            epoch_loss += float(loss.detach().cpu()) * batch["labels"].shape[0]
            epoch_logits.extend(outputs.logits.detach().cpu().flatten().tolist())
            gradient_norms.append(float(gradient_norm.detach().cpu()))

        metrics = {
            "epoch": epoch,
            "loss": round(epoch_loss / count, 6),
            "accuracy": round(correct / count, 6),
            "logits_min": round(min(epoch_logits), 6),
            "logits_max": round(max(epoch_logits), 6),
            "gradient_norm_mean": round(sum(gradient_norms) / len(gradient_norms), 6),
        }
        history.append(metrics)
        LOGGER.info(
            "epoch=%d/%d loss=%.6f accuracy=%.4f logits=[%.3f, %.3f] grad_norm=%.3f",
            epoch, EPOCHS, metrics["loss"], metrics["accuracy"],
            metrics["logits_min"], metrics["logits_max"], metrics["gradient_norm_mean"],
        )

    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_PATH)
    tokenizer.save_pretrained(CHECKPOINT_PATH)
    final = history[-1]
    report = {
        "model": MODEL_NAME,
        "intended_baseline_model": BASELINE_MODEL_NAME,
        "pretrained_weights": False,
        "device": str(device),
        "device_reason": device_reason,
        "cuda_available": cuda_available,
        "cuda_capability": cuda_capability,
        "seed": SEED,
        "rows": len(rows),
        "labels": {"0": ROWS_PER_LABEL, "1": ROWS_PER_LABEL},
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "history": history,
        "final_training_accuracy": final["accuracy"],
        "overfit_gate_passed": final["accuracy"] >= 0.95,
        "checkpoint": str(CHECKPOINT_PATH),
    }
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    LOGGER.info(
        "completed: final_training_accuracy=%.4f overfit_gate_passed=%s report=%s checkpoint=%s",
        final["accuracy"], report["overfit_gate_passed"], REPORT_PATH, CHECKPOINT_PATH,
    )
    return report


def main():
    configure_logging()
    try:
        rows = select_fixed_subset(read_rows(find_joined_file()))
        report = train(rows)
        print(json.dumps(report, indent=2))
    except Exception:
        LOGGER.exception("tiny overfit experiment failed")
        raise


if __name__ == "__main__":
    main()
