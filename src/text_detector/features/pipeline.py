"""
Feature Extraction Pipeline.
Combines Dynamic PMI Vocabulary scores and Statistical scores (Burstiness, RTTR, Entropy)
into a single unified feature vector for Machine Learning models.
"""

from typing import Dict, List
from src.text_detector.features.vocabulary_pmi import PMIFeatureExtractor
from src.text_detector.features.statistical_features import StatisticalFeatureExtractor


class FeaturePipeline:
    """Master pipeline for extracting all mathematical and linguistic features from text."""
    
    def __init__(self):
        # Initialize both feature extractors
        self.pmi_extractor = PMIFeatureExtractor()
        self.stats_extractor = StatisticalFeatureExtractor()

    def extract(self, text: str) -> Dict[str, float]:
        """
        Runs all extractors on a single piece of text and combines the results.
        Returns a dictionary of numerical features.
        """
        # 1. Get Vocabulary PMI score
        pmi_score = self.pmi_extractor.score_text(text)
        
        # 2. Get Statistical scores
        stats_scores = self.stats_extractor.score_text(text)
        
        # 3. Combine into a single feature vector
        features = {
            'pmi_score': pmi_score,
            'burstiness': stats_scores.get('burstiness', 0.0),
            'rttr': stats_scores.get('rttr', 0.0),
            'entropy': stats_scores.get('entropy', 0.0)
        }
        
        return features

    def extract_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Runs the extractor on a list of texts.
        Useful for processing large datasets efficiently.
        """
        return [self.extract(text) for text in texts]
