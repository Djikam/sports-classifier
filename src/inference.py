"""
Module d'inférence : prédiction sur une seule image.
"""
import numpy as np
from PIL import Image
from src.model import load_model
from src.utils import preprocess_image, decode_predictions, get_class_names


class SportsClassifier:
    """
    Wrapper pour le classificateur de sports.
    """
    def __init__(self):
        self.model = None
        self.class_names = None
        self._load()

    def _load(self):
        """Charge le modèle et les noms de classes."""
        self.model = load_model()
        self.class_names = get_class_names()
        if not self.class_names:
            # Fallback si pas de données locales
            self.class_names = [f"Sport_{i}" for i in range(100)]

    def predict(self, image, top_k=5):
        """
        Prédit la classe d'une image.

        Args:
            image: PIL Image ou chemin vers image
            top_k: nombre de prédictions à retourner

        Returns:
            liste de tuples (nom_classe, probabilité)
        """
        if isinstance(image, str):
            image = Image.open(image)

        # Preprocessing
        img_array = preprocess_image(image)

        # Prédiction
        predictions = self.model.predict(img_array, verbose=0)

        # Décodage
        results = decode_predictions(predictions, self.class_names, top_k=top_k)

        return results

    def predict_class(self, image):
        """
        Retourne uniquement la classe la plus probable.

        Returns:
            tuple (nom_classe, probabilité)
        """
        results = self.predict(image, top_k=1)
        return results[0]
