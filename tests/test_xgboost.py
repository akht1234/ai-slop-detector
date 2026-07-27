"""
Trains and verifies the Local XGBoost Baseline on the Wikipedia Human-AI dataset.
"""

import os
import sys

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.data.download_dataset import load_parsed_dataset
from src.text_detector.models.xgboost_classifier import XGBoostBaselineDetector

def main():
    print("--- 1. LOADING DATASET ---")
    samples = load_parsed_dataset()
    
    # Take a small subset for fast local training test (e.g. 4000 rows)
    samples = samples[:4000]
    texts = [s['text'] for s in samples]
    labels = [s['label'] for s in samples]
    
    # 80% Train, 20% Test Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print(f"Train Set: {len(train_texts)} samples")
    print(f"Test Set:  {len(test_texts)} samples")

    print("\n--- 2. TRAINING FAST XGBOOST BASELINE ---")
    detector = XGBoostBaselineDetector()
    detector.train(train_texts, train_labels)

    print("\n--- 3. EVALUATING ACCURACY ---")
    # Predict probabilities for the test set
    predictions = []
    # (In a real scenario we'd vectorize this, but a simple loop is fine for the test script)
    for t in test_texts:
        prob = detector.predict_proba(t)
        # Threshold at 0.5
        predictions.append(1 if prob > 0.5 else 0)
        
    accuracy = accuracy_score(test_labels, predictions)
    print(f"XGBoost Test Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(test_labels, predictions, target_names=["Human (0)", "AI (1)"]))

    print("\n--- 4. FEATURE IMPORTANCE (EXPLAINABILITY) ---")
    importance = detector.get_feature_importance()
    for name, imp in importance.items():
        print(f"{name:12s}: {imp*100:.1f}%")

    print("\n--- 5. SAVING MODEL ---")
    detector.save()

if __name__ == "__main__":
    main()
