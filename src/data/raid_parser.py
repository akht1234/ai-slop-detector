"""
Official RAID Dataset Parser & Inspector.
Loads, parses, and cleans records from local RAID CSV files (e.g. data/raid_sample.csv).
"""

import os
import csv
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RAIDSample:
    text: str
    model: str          # e.g., 'gpt4', 'llama2', 'mistral', 'human'
    domain: str         # e.g., 'news', 'abstracts', 'creative', 'recipes'
    attack: str         # e.g., 'none', 'synonym', 'back_translation'
    decoding: str       # e.g., 'greedy', 'sampling'
    title: str = ""
    is_ai: int = 1      # 1 = AI, 0 = Human


def parse_local_raid_csv(file_path: str, max_rows: Optional[int] = None) -> List[RAIDSample]:
    """
    Parses a local RAID CSV file into clean RAIDSample objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RAID file not found at: {file_path}")

    samples = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            
            text = row.get("generation") or row.get("text") or ""
            model = row.get("model") or "unknown"
            domain = row.get("domain") or "general"
            attack = row.get("attack") or "none"
            decoding = row.get("decoding") or "default"
            title = row.get("title") or ""
            
            is_ai = 0 if model.lower() == "human" else 1

            if text.strip():
                samples.append(RAIDSample(
                    text=text.strip(),
                    model=model,
                    domain=domain,
                    attack=attack,
                    decoding=decoding,
                    title=title,
                    is_ai=is_ai
                ))

    return samples


def display_raid_summary(samples: List[RAIDSample]) -> None:
    """
    Prints a detailed distribution summary of the loaded RAID dataset slice.
    """
    total = len(samples)
    ai_count = sum(1 for s in samples if s.is_ai == 1)
    human_count = total - ai_count

    domains = {}
    models = {}
    attacks = {}

    for s in samples:
        domains[s.domain] = domains.get(s.domain, 0) + 1
        models[s.model] = models.get(s.model, 0) + 1
        attacks[s.attack] = attacks.get(s.attack, 0) + 1

    print("\n" + "=" * 65)
    print(" OFFICIAL RAID BENCHMARK PARSED SUMMARY")
    print("=" * 65)
    print(f" Total Samples Parsed: {total}")
    print(f" Human Samples:        {human_count} ({(human_count/total)*100:.1f}%)" if total else "")
    print(f" AI Slop Samples:      {ai_count} ({(ai_count/total)*100:.1f}%)" if total else "")
    print("-" * 65)
    print(f" Domains Found ({len(domains)}):", dict(sorted(domains.items(), key=lambda x: x[1], reverse=True)))
    print(f" Models Found ({len(models)}): ", dict(sorted(models.items(), key=lambda x: x[1], reverse=True)))
    print(f" Attacks Found ({len(attacks)}):", dict(sorted(attacks.items(), key=lambda x: x[1], reverse=True)))
    print("=" * 65)
