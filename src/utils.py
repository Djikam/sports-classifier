"""
Fonctions utilitaires pour le projet Sports Classifier.
"""
import os
import json
import numpy as np
from PIL import Image

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "sports")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Paramètres
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 100


def get_class_names():
    """
    Retourne la liste des noms de classes (sports).
    À adapter selon le dataset utilisé.
    """
    # Si vous avez un fichier CSV ou JSON avec les classes
    classes_path = os.path.join(DATA_DIR, "class_dict.csv")
    if os.path.exists(classes_path):
        import pandas as pd
        df = pd.read_csv(classes_path)
        return sorted(df['class'].unique().tolist())

    # Fallback : extraire des dossiers
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.exists(train_dir):
        return sorted(os.listdir(train_dir))

    return []


def preprocess_image(image, target_size=(224, 224)):
    """
    Prétraite une image PIL pour le modèle.

    Args:
        image: PIL Image
        target_size: tuple (H, W)

    Returns:
        numpy array normalisé [1, H, W, 3]
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def decode_predictions(predictions, class_names, top_k=5):
    """
    Décodage des prédictions en labels lisibles.

    Args:
        predictions: array de probabilités [1, num_classes]
        class_names: liste des noms de classes
        top_k: nombre de top prédictions à retourner

    Returns:
        liste de tuples (nom_classe, probabilité)
    """
    predictions = np.squeeze(predictions)
    top_indices = np.argsort(predictions)[::-1][:top_k]

    results = []
    for idx in top_indices:
        class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
        prob = float(predictions[idx])
        results.append((class_name, prob))

    return results
