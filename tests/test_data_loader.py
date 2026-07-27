"""
Simple script to test loading and inspecting dataset samples.
"""

import sys
import os

# Add src to python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.raid_loader import RAIDDataLoader, inspect_dataset_samples


def main():
    print("Testing RAID Data Loader...")
    loader = RAIDDataLoader()
    
    # Load fallback samples for immediate local inspection
    df = loader.load_fallback_samples()
    print(f"Loaded {len(df)} samples successfully.")
    
    # Inspect samples
    inspect_dataset_samples(df, n_samples=2)


if __name__ == "__main__":
    main()
