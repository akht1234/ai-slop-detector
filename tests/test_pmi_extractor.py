"""
Verification script to test the Dynamic PMI AI Vocabulary Builder on real RAID data.
"""

import os
import sys

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.download_dataset import load_parsed_dataset
from src.text_detector.features.vocabulary_pmi import PMIVocabularyBuilder, PMIFeatureExtractor

def main():
    print("--- 1. LOADING DATASET ---")
    
    samples = load_parsed_dataset()
    
    human_texts = [s['text'] for s in samples if s['label'] == 0]
    ai_texts = [s['text'] for s in samples if s['label'] == 1]

    
    print(f"Loaded {len(human_texts)} Human texts and {len(ai_texts)} AI texts.")

    print("\n--- 2. TRAINING DYNAMIC PMI VOCABULARY ---")
    builder = PMIVocabularyBuilder(min_freq=5, top_k=500)
    builder.fit(human_texts, ai_texts)
    builder.save()

    print("\n--- 3. TOP 20 AI BUZZWORDS DISCOVERED DYNAMICALLY ---")
    # Sort the dictionary by score
    top_words = sorted(builder.ai_vocab_weights.items(), key=lambda x: x[1], reverse=True)[:20]
    for i, (word, score) in enumerate(top_words, 1):
        print(f"{i:2d}. {word:15s} (PMI: +{score:.2f})")

    print("\n--- 4. TESTING INFERENCE SCORER ---")
    scorer = PMIFeatureExtractor()
    
    if human_texts:
        h_score = scorer.score_text(human_texts[0])
        print(f"Human Text #1 PMI Density Score: {h_score:.4f}")
    if ai_texts:
        a_score = scorer.score_text(ai_texts[0])
        print(f"AI Text #1 PMI Density Score:    {a_score:.4f}")


if __name__ == "__main__":
    main()
