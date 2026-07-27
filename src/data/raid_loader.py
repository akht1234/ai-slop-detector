"""
Full RAID Benchmark Dataset Loader (16 GB Dataset Support).
Supports streaming and batch loading from 'liamdugan/raid' on Hugging Face.
"""

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional
import os


@dataclass
class RAIDRecord:
    text: str
    label: int          # 0 = Human, 1 = AI Slop
    model: str          # e.g., 'gpt4', 'llama2', 'mistral', 'claude', 'human'
    domain: str         # e.g., 'news', 'abstracts', 'creative', 'recipes', 'reddit'
    attack: str         # e.g., 'none', 'synonym', 'back_translation', 'homoglyph'
    title: str = ""


class FullRAIDLoader:
    """
    Handles streaming and partition loading for the full ~16 GB RAID benchmark.
    HuggingFace repo: 'liamdugan/raid'
    """

    def __init__(self, dataset_name: str = "liamdugan/raid"):
        self.dataset_name = dataset_name

    def stream_dataset(self, split: str = "train", limit: Optional[int] = 100) -> Generator[RAIDRecord, None, None]:
        """
        Streams records line-by-line without downloading the entire 16 GB file at once.
        """
        try:
            from datasets import load_dataset
            print(f"[RAID Loader] Streaming from '{self.dataset_name}' (split='{split}')...")
            ds = load_dataset(self.dataset_name, split=split, streaming=True)
            
            count = 0
            for row in ds:
                if limit and count >= limit:
                    break
                text = row.get("generation") or row.get("text") or row.get("content", "")
                raw_label = row.get("label") or row.get("is_ai", 0)
                label = 1 if "1" in str(raw_label) or "ai" in str(raw_label).lower() else 0
                model = row.get("model") or row.get("generator") or "unknown"
                domain = row.get("domain", "general")
                attack = row.get("attack", "none")
                title = row.get("title", "")

                yield RAIDRecord(
                    text=text,
                    label=label,
                    model=model,
                    domain=domain,
                    attack=attack,
                    title=title
                )
                count += 1
        except Exception as e:
            print(f"[RAID Loader Error] Streaming failed: {e}")
            print("[RAID Loader] Ensure 'datasets' package is installed: pip install datasets")

    def inspect_raid_stream(self, split: str = "train", n_samples: int = 4) -> List[RAIDRecord]:
        """
        Inspects the first n_samples from the 16 GB dataset stream.
        """
        records = list(self.stream_dataset(split=split, limit=n_samples))
        print("\n" + "=" * 70)
        print(f" FULL RAID BENCHMARK STREAM INSPECTION ({len(records)} Samples)")
        print("=" * 70)
        for i, r in enumerate(records, 1):
            label_str = "AI SLOP (1)" if r.label == 1 else "HUMAN (0)"
            print(f"\nSample #{i} [{label_str}] | Domain: {r.domain} | Model: {r.model} | Attack: {r.attack}")
            print("-" * 70)
            print(f"Text: \"{r.text[:300]}...\"")
        print("=" * 70)
        return records
