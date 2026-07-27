"""
Verification test for local RAID sample dataset parser.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.raid_parser import parse_local_raid_csv, display_raid_summary

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raid_sample.csv"))


def main():
    print(f"Loading local RAID sample from: {DATA_PATH}")
    samples = parse_local_raid_csv(DATA_PATH, max_rows=1000)
    print(f"Successfully loaded {len(samples)} RAID text records!")
    
    print("\n--- SAMPLE RECORD #1 ---")
    s1 = samples[0]
    print(f"ID/Model:  {s1.model}")
    print(f"Domain:    {s1.domain}")
    print(f"Text Length: {len(s1.text)} chars")
    print(f"Text Snippet:\n{s1.text[:300]}...")


if __name__ == "__main__":
    main()
