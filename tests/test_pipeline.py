"""
Verification test for the Feature Extraction Pipeline.
Ensures both PMI and Statistical features are combined successfully.
"""

import os
import sys

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.download_dataset import load_parsed_dataset
from src.text_detector.features.pipeline import FeaturePipeline

def main():
    print("--- 1. INITIALIZING PIPELINE ---")
    pipeline = FeaturePipeline()
    print("Pipeline initialized successfully with PMI and Statistical extractors.")

    print("\n--- 2. LOADING DATASET ---")
    samples = load_parsed_dataset()
    human_texts = [s['text'] for s in samples if s['label'] == 0]
    ai_texts = [s['text'] for s in samples if s['label'] == 1]

    print("\n--- 3. EXTRACTING COMBINED FEATURES ---")
    
    print("\n[HUMAN SAMPLE]")
    h_features = pipeline.extract(human_texts[0])
    for k, v in h_features.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n[AI SAMPLE]")
    a_features = pipeline.extract(ai_texts[0])
    for k, v in a_features.items():
        print(f"  {k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()
