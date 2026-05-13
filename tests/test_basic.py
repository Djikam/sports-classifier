"""
Tests basiques pour le projet Sports Classifier.
"""
import os
import sys
import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import preprocess_image, decode_predictions


def test_preprocess_image():
    """Test le preprocessing d'une image."""
    img = Image.new('RGB', (300, 400), color='red')
    processed = preprocess_image(img, target_size=(224, 224))

    assert processed.shape == (1, 224, 224, 3)
    assert np.max(processed) <= 1.0
    assert np.min(processed) >= 0.0


def test_decode_predictions():
    """Test le décodage des prédictions."""
    preds = np.zeros((1, 100))
    preds[0, 5] = 0.8
    preds[0, 10] = 0.15
    preds[0, 20] = 0.05

    class_names = [f"Sport_{i}" for i in range(100)]
    results = decode_predictions(preds, class_names, top_k=3)

    assert len(results) == 3
    assert results[0][0] == "Sport_5"
    assert results[0][1] == 0.8


if __name__ == "__main__":
    test_preprocess_image()
    test_decode_predictions()
    print("✅ Tous les tests ont réussi !")
