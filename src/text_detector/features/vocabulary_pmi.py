"""
Dynamic AI Vocabulary Detection using Pointwise Mutual Information (PMI).
Learns AI buzzwords automatically from a dataset without hardcoding them.
"""

import os
import json
import math
import re
from collections import Counter
from typing import List, Dict, Tuple

# Path to save the dynamically learned vocabulary weights
VOCAB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "ai_vocabulary_weights.json"))


def tokenize(text: str) -> List[str]:
    """Simple regex tokenizer that lowers text and extracts alphanumeric words."""
    # Convert to lowercase and extract words
    return re.findall(r'\b[a-z]{3,}\b', text.lower())


class PMIVocabularyBuilder:
    """Builds a dynamic dictionary of AI buzzwords using PMI from a training corpus."""
    
    def __init__(self, min_freq: int = 10, top_k: int = 1000):
        self.min_freq = min_freq
        self.top_k = top_k
        self.ai_vocab_weights: Dict[str, float] = {}

    def fit(self, human_texts: List[str], ai_texts: List[str]) -> None:
        """Calculates PMI scores for all words comparing AI vs Human text."""
        print("[PMI Builder] Tokenizing human texts...")
        human_counter = Counter()
        for text in human_texts:
            human_counter.update(tokenize(text))

        print("[PMI Builder] Tokenizing AI texts...")
        ai_counter = Counter()
        for text in ai_texts:
            ai_counter.update(tokenize(text))

        # Total words in each corpus
        total_human_words = sum(human_counter.values())
        total_ai_words = sum(ai_counter.values())

        if total_human_words == 0 or total_ai_words == 0:
            raise ValueError("Empty corpus provided to PMI Builder!")

        print(f"[PMI Builder] Corpus Size -> Human: {total_human_words} words, AI: {total_ai_words} words")

        # Calculate PMI for each word
        # Formula: PMI(word, AI) = log ( P(word | AI) / P(word | Human) )
        pmi_scores: Dict[str, float] = {}
        
        # We only consider words that appear at least `min_freq` times across BOTH corpora
        # to filter out ultra-rare noise.
        all_words = set(human_counter.keys()).union(set(ai_counter.keys()))
        
        for word in all_words:
            total_freq = human_counter[word] + ai_counter[word]
            if total_freq < self.min_freq:
                continue
            
            # Laplace Smoothing (+1) to avoid probability of 0
            # P(word | Human)
            p_human = (human_counter.get(word, 0) + 1) / (total_human_words + len(all_words))
            # P(word | AI)
            p_ai = (ai_counter.get(word, 0) + 1) / (total_ai_words + len(all_words))
            
            # PMI ratio: How much more likely is this word in AI vs Human?
            # We use log2 for readability.
            pmi = math.log2(p_ai / p_human)
            pmi_scores[word] = pmi

        # We want the words most strongly associated with AI.
        # Sort by PMI score descending
        sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Save the top K AI buzzwords
        self.ai_vocab_weights = {word: score for word, score in sorted_pmi[:self.top_k] if score > 0}
        print(f"[PMI Builder] Successfully learned {len(self.ai_vocab_weights)} AI-specific buzzwords!")

    def save(self, filepath: str = VOCAB_FILE) -> None:
        """Saves the learned PMI weights to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.ai_vocab_weights, f, indent=2)
        print(f"[PMI Builder] Saved dynamic AI vocabulary to {filepath}")


class PMIFeatureExtractor:
    """Uses a pre-trained PMI vocabulary to score new text."""
    
    def __init__(self, filepath: str = VOCAB_FILE):
        self.ai_vocab_weights: Dict[str, float] = {}
        self.load(filepath)

    def load(self, filepath: str) -> None:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.ai_vocab_weights = json.load(f)
        else:
            print(f"[Warning] PMI vocabulary file not found at {filepath}. Call PMIVocabularyBuilder.fit() first.")

    def score_text(self, text: str) -> float:
        """
        Calculates the total AI PMI score of a text.
        Higher positive score = Strong AI signal.
        """
        if not self.ai_vocab_weights:
            return 0.0
            
        words = tokenize(text)
        if not words:
            return 0.0
            
        total_pmi = 0.0
        ai_words_found = 0
        
        for word in words:
            if word in self.ai_vocab_weights:
                total_pmi += self.ai_vocab_weights[word]
                ai_words_found += 1
                
        # Normalize by length to avoid penalizing long texts
        density_score = total_pmi / len(words)
        return density_score
