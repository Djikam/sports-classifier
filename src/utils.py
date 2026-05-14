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
    Ordre alphabétique du dataset Kaggle 100 Sports.
    """
    # LISTE HARDCODÉE DES 100 SPORTS (ordre alphabétique)
    # Cette liste est TOUJOURS retournée, même si les fichiers locaux n'existent pas
    SPORTS_LIST = [
        "air hockey", "amateur wrestling", "american football", "archery",
        "arm wrestling", "axe throwing", "balance beam", "barrel racing",
        "baseball", "basketball", "billiards", "bmx", "bobsled", "bowling",
        "boxing", "bull riding", "canoe slalom", "cheerleading", "cliff diving",
        "cricket", "croquet", "curling", "disc golf", "fencing", "field hockey",
        "figure skating pairs", "figure skating women", "fly fishing",
        "football", "formula 1 racing", "frisbee", "gaga ball", "golf",
        "hammer throw", "hang gliding", "harness racing", "high jump",
        "hockey", "horse jumping", "horse racing", "horseshoe pitching",
        "hurdles", "hydroplane racing", "ice climbing", "ice yachting",
        "jai alai", "javelin", "judo", "lacrosse", "luge", "motorcycle racing",
        "mushing", "nascar racing", "olympic wrestling", "parallel bars",
        "pole dancing", "pole vault", "polo", "pommel horse", "rings",
        "rock climbing", "roller derby", "rollerblade racing", "rowing",
        "rugby", "sailboat racing", "shot put", "shuffleboard", "ski jumping",
        "sky surfing", "snowboarding", "snowmobile racing", "speed skating",
        "squash", "sumo wrestling", "surfing", "swimming", "table tennis",
        "tennis", "track bicycle", "trapeze", "tug of war", "ultimate frisbee",
        "uneven bars", "volleyball", "water cycling", "water polo",
        "weightlifting", "wheelchair basketball", "wheelchair racing",
        "wingsuit flying"
    ]
    
    # Vérifie d'abord si un fichier local existe (priorité aux données réelles)
    classes_path = os.path.join(DATA_DIR, "class_dict.csv")
    if os.path.exists(classes_path):
        try:
            import pandas as pd
            df = pd.read_csv(classes_path)
            return sorted(df['class'].unique().tolist())
        except Exception as e:
            print(f"⚠️ Erreur lecture CSV: {e}")
    
    # Essaie les dossiers d'entraînement
    train_dir = os.path.join(DATA_DIR, "train")
    if os.path.exists(train_dir):
        try:
            return sorted(os.listdir(train_dir))
        except Exception as e:
            print(f"⚠️ Erreur lecture dossier: {e}")
    
    # Fallback : retourne la liste hardcodée (pour Render/production)
    print(f"⚠️ Utilisation liste hardcodée ({len(SPORTS_LIST)} sports)")
    return SPORTS_LIST


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