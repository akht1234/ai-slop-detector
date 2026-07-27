"""
Dataset Downloader & Inspector: Downloads and parses the full Wikipedia Human vs AI dataset
(9,970 paired paragraph records from gouwsxander/slop-detector).
"""

import os
import sys
import json
import urllib.request
from typing import Dict, List

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
OUTPUT_FILE = os.path.join(DATASET_DIR, "wikipedia_human_ai.jsonl")

# Raw GitHub URL for gouwsxander/slop-detector dataset
PRIMARY_URL = "https://raw.githubusercontent.com/gouwsxander/slop-detector/main/data/wikipedia.jsonl"
FALLBACK_HF_URL = "https://huggingface.co/datasets/gouwsxander/wikipedia-human-ai/raw/main/wikipedia.jsonl"


def download_full_dataset(url: str = PRIMARY_URL, output_path: str = OUTPUT_FILE) -> str:
    """
    Downloads the full JSONL dataset from GitHub or HuggingFace.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[Dataset] Full dataset already exists locally at: {output_path}")
        print(f"[Dataset] File Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        return output_path

    print(f"[Dataset] Downloading full dataset from: {url} ...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            block_size = 8192
            downloaded = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                print(f"\rDownloaded: {downloaded / (1024*1024):.2f} MB", end="", flush=True)
            print()
        print(f"[Dataset] Download completed successfully -> {output_path}")
        return output_path
    except Exception as e:
        print(f"\n[Dataset Error] Failed downloading from primary URL: {e}")
        if url != FALLBACK_HF_URL:
            print("[Dataset] Trying fallback HuggingFace URL...")
            return download_full_dataset(url=FALLBACK_HF_URL, output_path=output_path)
        raise e


def load_parsed_dataset(file_path: str = OUTPUT_FILE) -> List[Dict]:
    """
    Parses the JSONL dataset into flattened single-label records:
    [
        {'text': ..., 'label': 0, 'page_title': ...},  # Human
        {'text': ..., 'label': 1, 'page_title': ...}   # AI
    ]
    """
    if not os.path.exists(file_path):
        download_full_dataset(output_path=file_path)

    flattened = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                title = item.get("page_title", "Unknown")
                
                # Human paragraph (Label = 0)
                if "human_text" in item and item["human_text"]:
                    flattened.append({
                        "text": item["human_text"].strip(),
                        "label": 0,
                        "page_title": title,
                        "model": "human"
                    })
                
                # AI rewrite paragraph (Label = 1)
                if "ai_text" in item and item["ai_text"]:
                    flattened.append({
                        "text": item["ai_text"].strip(),
                        "label": 1,
                        "page_title": title,
                        "model": "gpt_rewrite"
                    })
    return flattened


def inspect_full_dataset(file_path: str = OUTPUT_FILE, max_records: int = 2) -> None:
    """
    Reads and summarizes records from the downloaded dataset.
    """
    parsed = load_parsed_dataset(file_path)
    human_records = [r for r in parsed if r["label"] == 0]
    ai_records = [r for r in parsed if r["label"] == 1]

    print("\n" + "=" * 65)
    print(f" FULL DATASET SUMMARY: {len(parsed)} Total Paragraphs ({len(parsed)//2} Pairs)")
    print("=" * 65)
    print(f" Total Human Paragraphs: {len(human_records)}")
    print(f" Total AI Paragraphs:    {len(ai_records)}")

    print("\n" + "=" * 65)
    print(" [HUMAN PARAGRAPH SAMPLES] (Label = 0)")
    print("=" * 65)
    for sample in human_records[:max_records]:
        print(f"[Topic: {sample['page_title']}]")
        print(f"\"{sample['text']}\"\n")

    print("=" * 65)
    print(" [AI REWRITE PARAGRAPH SAMPLES] (Label = 1)")
    print("=" * 65)
    for sample in ai_records[:max_records]:
        print(f"[Topic: {sample['page_title']} | Model: {sample['model']}]")
        print(f"\"{sample['text']}\"\n")
    print("=" * 65)


if __name__ == "__main__":
    download_full_dataset()
    inspect_full_dataset()
