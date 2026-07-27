"""
Verification test for Statistical Features (Burstiness, RTTR, Entropy).
"""

import os
import sys

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.download_dataset import load_parsed_dataset
from src.text_detector.features.statistical_features import StatisticalFeatureExtractor

def main():
    print("--- 1. LOADING DATASET ---")
    samples = load_parsed_dataset()
    human_texts = [s['text'] for s in samples if s['label'] == 0]
    ai_texts = [s['text'] for s in samples if s['label'] == 1]

    print("\n--- 2. EXTRACTING STATISTICAL FEATURES ---")
    extractor = StatisticalFeatureExtractor()
    
    print("\n[HUMAN SAMPLE]")
    h_text = human_texts[0]
    h_scores = extractor.score_text(h_text)
    print(f"Text Snippet: \"{h_text[:100]}...\"")
    print(f"Burstiness (CV): {h_scores['burstiness']:.4f}")
    print(f"Lexical (RTTR):  {h_scores['rttr']:.4f}")
    print(f"Entropy:         {h_scores['entropy']:.4f}")

    print("\n[AI SAMPLE]")
    a_text = ai_texts[0]
    a_scores = extractor.score_text(a_text)
    print(f"Text Snippet: \"{a_text[:100]}...\"")
    print(f"Burstiness (CV): {a_scores['burstiness']:.4f}")
    print(f"Lexical (RTTR):  {a_scores['rttr']:.4f}")
    print(f"Entropy:         {a_scores['entropy']:.4f}")

    # Calculate average across 1000 samples to prove statistical separation
    print("\n--- 3. POPULATION AVERAGES (1000 Samples) ---")
    
    h_avg = {'burstiness': 0, 'rttr': 0, 'entropy': 0}
    a_avg = {'burstiness': 0, 'rttr': 0, 'entropy': 0}
    
    N = 1000
    for i in range(N):
        s1 = extractor.score_text(human_texts[i])
        s2 = extractor.score_text(ai_texts[i])
        for k in h_avg:
            h_avg[k] += s1[k]
            a_avg[k] += s2[k]
            
    print("Average Human Scores:")
    for k in h_avg: print(f"  {k:12s}: {h_avg[k]/N:.4f}")
        
    print("Average AI Scores:")
    for k in a_avg: print(f"  {k:12s}: {a_avg[k]/N:.4f}")


if __name__ == "__main__":
    main()
