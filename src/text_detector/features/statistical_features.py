"""
Statistical Feature Extractor.
Extracts mathematical signatures (Burstiness, Lexical Diversity, Shannon Entropy) from text.
"""

import math
import re
from collections import Counter
import statistics
from typing import Dict


class StatisticalFeatureExtractor:
    
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Split by ., !, or ? followed by whitespace
        sentences = re.split(r'[.!?]+(?:\s+|$)', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\b[a-z]+\b', text.lower())

    def calculate_burstiness(self, text: str) -> float:
        """
        Calculates the coefficient of variation (sigma / mu) of sentence lengths.
        High = Human (bursty), Low = AI (monotonic).
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0
            
        # Count words per sentence
        lengths = [len(self._tokenize(s)) for s in sentences]
        lengths = [l for l in lengths if l > 0]
        
        if len(lengths) < 2:
            return 0.0  # Cannot calculate variance of 1 sentence
            
        mu = statistics.mean(lengths)
        if mu == 0:
            return 0.0
            
        sigma = statistics.pstdev(lengths)
        return sigma / mu

    def calculate_lexical_diversity(self, text: str) -> float:
        """
        Calculates Root Type-Token Ratio (RTTR).
        RTTR = Unique Words / sqrt(Total Words)
        """
        words = self._tokenize(text)
        total_tokens = len(words)
        if total_tokens == 0:
            return 0.0
            
        unique_types = len(set(words))
        return unique_types / math.sqrt(total_tokens)

    def calculate_shannon_entropy(self, text: str) -> float:
        """
        Calculates Shannon Entropy of word frequencies.
        H = -sum( p(x) * log2(p(x)) )
        """
        words = self._tokenize(text)
        total_tokens = len(words)
        if total_tokens == 0:
            return 0.0
            
        counts = Counter(words)
        entropy = 0.0
        
        for count in counts.values():
            p_x = count / total_tokens
            entropy -= p_x * math.log2(p_x)
            
        return entropy

    def score_text(self, text: str) -> Dict[str, float]:
        """Returns all statistical features as a dictionary."""
        return {
            'burstiness': self.calculate_burstiness(text),
            'rttr': self.calculate_lexical_diversity(text),
            'entropy': self.calculate_shannon_entropy(text)
        }
