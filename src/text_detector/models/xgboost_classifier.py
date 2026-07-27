"""
Fast Local Baseline Model using XGBoost.
Trains a gradient-boosted decision tree on the 4 statistical/linguistic features.
"""

import os
import json
import numpy as np
import xgboost as xgb
from typing import List, Dict, Tuple
from src.text_detector.features.pipeline import FeaturePipeline

MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "xgb_baseline.json"))


class XGBoostBaselineDetector:
    
    def __init__(self):
        self.pipeline = FeaturePipeline()
        # Initialize the XGBoost model with safe, robust hyperparameters
        self.model = xgb.XGBClassifier(
            n_estimators=200,          # Number of trees
            learning_rate=0.05,        # Small steps to avoid overfitting
            max_depth=4,               # Shallow trees for better generalization
            subsample=0.8,             # Randomly sample 80% of data per tree
            colsample_bytree=0.8,      # Randomly sample 80% of features per tree
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
        self.is_trained = False

    def _texts_to_matrix(self, texts: List[str]) -> np.ndarray:
        """Converts raw texts into an N x 4 feature matrix."""
        print(f"[XGBoost] Extracting features for {len(texts)} samples...")
        features_list = self.pipeline.extract_batch(texts)
        
        # Ensure exact column order: [pmi_score, burstiness, rttr, entropy]
        matrix = []
        for f in features_list:
            row = [
                f.get('pmi_score', 0.0),
                f.get('burstiness', 0.0),
                f.get('rttr', 0.0),
                f.get('entropy', 0.0)
            ]
            matrix.append(row)
            
        return np.array(matrix)

    def train(self, texts: List[str], labels: List[int]) -> None:
        """Trains the XGBoost model on a set of texts and binary labels (0=Human, 1=AI)."""
        X = self._texts_to_matrix(texts)
        y = np.array(labels)
        
        print(f"[XGBoost] Training model on matrix shape {X.shape}...")
        self.model.fit(X, y)
        self.is_trained = True
        print("[XGBoost] Training complete!")

    def predict_proba(self, text: str) -> float:
        """Returns the probability (0.0 to 1.0) that the text is AI Slop."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
            
        X_single = self._texts_to_matrix([text])
        # predict_proba returns [[P(Human), P(AI)]]
        ai_probability = self.model.predict_proba(X_single)[0][1]
        return float(ai_probability)

    def get_feature_importance(self) -> Dict[str, float]:
        """Returns the importance of each feature in the decision trees."""
        if not self.is_trained:
            return {}
            
        names = ['pmi_score', 'burstiness', 'rttr', 'entropy']
        importances = self.model.feature_importances_
        
        importance_dict = {name: float(imp) for name, imp in zip(names, importances)}
        # Sort descending
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    def save(self, filepath: str = MODEL_SAVE_PATH) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        print(f"[XGBoost] Model saved to {filepath}")

    def load(self, filepath: str = MODEL_SAVE_PATH) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No XGBoost model found at {filepath}")
        self.model.load_model(filepath)
        self.is_trained = True
        print(f"[XGBoost] Model loaded from {filepath}")
