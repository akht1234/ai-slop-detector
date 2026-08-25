"""
Downloads a real 50 MB slice of the official RAID Benchmark dataset ('liamdugan/raid/train.csv')
directly from Hugging Face using HTTP Range request.
"""

import os
import sys
import csv
import io
import urllib.request

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
RAID_CSV_FILE = os.path.join(DATA_DIR, "raid_sample.csv")

# Direct HuggingFace URLs in liamdugan/raid.
TRAIN_CSV_URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
TEST_CSV_URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/test.csv"
RAID_TRAIN_CSV_FILE = os.path.join(DATA_DIR, "raid_train_sample.csv")
RAID_TEST_CSV_FILE = os.path.join(DATA_DIR, "raid_sample.csv")
BYTES_TO_DOWNLOAD = 50 * 1024 * 1024  # 50 MB slice


def download_raid_subset(
    output_path: str = RAID_TRAIN_CSV_FILE,
    target_bytes: int = BYTES_TO_DOWNLOAD,
    split: str = "train",
) -> str:
    """
    Downloads a bounded byte slice of a RAID split directly from Hugging Face.

    The train split is the default because it contains labels/metadata needed
    for training. The test split is intended for unlabeled evaluation.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'")

    source_url = TRAIN_CSV_URL if split == "train" else TEST_CSV_URL

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
        print(f"[RAID Downloader] RAID subset already exists at: {output_path}")
        print(f"[RAID Downloader] File Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        return output_path

    print(f"[RAID Downloader] Fetching {target_bytes / (1024 * 1024):.0f} MB slice from official RAID benchmark ('{split}.csv')...")
    print(f"[RAID Downloader] Source: {source_url}")

    try:
        req = urllib.request.Request(
            source_url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'Range': f'bytes=0-{target_bytes - 1}'
            }
        )
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            block_size = 65536
            downloaded = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                remaining = target_bytes - downloaded
                if remaining <= 0:
                    break
                buffer = buffer[:remaining]
                downloaded += len(buffer)
                out_file.write(buffer)
                mb = downloaded / (1024 * 1024)
                print(f"\rDownloading RAID Slice: {mb:.2f} MB / {target_bytes/(1024*1024):.1f} MB", end="", flush=True)
                if downloaded >= target_bytes:
                    break
            print()
        print(f"[RAID Downloader] Download complete -> {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")
        return output_path
    except Exception as e:
        print(f"\n[RAID Downloader Error]: {e}")
        raise e


def inspect_raid_file(file_path: str = RAID_TRAIN_CSV_FILE, max_records: int = 2) -> None:
    """
    Inspects and parses records from the downloaded RAID CSV slice.
    """
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return

    records = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            records.append(row)
            if i >= 5000:
                break

    print("\n" + "=" * 65)
    print(" OFFICIAL RAID BENCHMARK DATASET SLICE INSPECTION")
    print("=" * 65)
    print(f" Total Rows Parsed: {len(records)}")
    if records:
        print(" CSV Columns:", list(records[0].keys()))

    human_samples = [r for r in records if r.get("model") == "human" or r.get("label") == "0"][:max_records]
    ai_samples = [r for r in records if r.get("model") != "human" or r.get("label") == "1"][:max_records]

    print("\n" + "=" * 65)
    print(" [HUMAN SAMPLE FROM RAID]")
    print("=" * 65)
    if human_samples:
        s = human_samples[0]
        text_val = s.get("generation") or s.get("text") or str(s)
        print(f"[Domain: {s.get('domain')} | Model: {s.get('model')}]")
        print(f"\"{text_val[:300]}...\"\n")

    print("=" * 65)
    print(" [AI SLOP SAMPLE FROM RAID]")
    print("=" * 65)
    if ai_samples:
        s = ai_samples[0]
        text_val = s.get("generation") or s.get("text") or str(s)
        print(f"[Domain: {s.get('domain')} | Model: {s.get('model')} | Attack: {s.get('attack')}]")
        print(f"\"{text_val[:300]}...\"\n")
    print("=" * 65)


if __name__ == "__main__":
    download_file = download_raid_subset()
    inspect_raid_file(download_file)
