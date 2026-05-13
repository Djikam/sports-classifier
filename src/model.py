"""
Gestion du modèle : architecture, chargement, téléchargement.
"""
import os
import gdown
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from src.utils import MODELS_DIR, NUM_CLASSES, IMG_SIZE

# URL Google Drive - fichier au format .h5 (HDF5)
MODEL_DRIVE_URL = os.environ.get(
    "MODEL_DRIVE_URL",
    "https://drive.google.com/uc?id=1S46qADbZHvMNOf13_1-WLvhLGnY4RqN-&export=download&confirm=t"
)

# Chemin local du modèle (format HDF5 compatible TF 2.x)
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")


def diagnose_model_file(path):
    """Diagnostic complet du fichier modèle."""
    if not os.path.exists(path):
        print("❌ Fichier inexistant")
        return None

    size = os.path.getsize(path)
    print(f"📁 Fichier : {path}")
    print(f"📊 Taille : {size / 1024 / 1024:.2f} MB")

    with open(path, 'rb') as f:
        header = f.read(8)

    print(f"🔍 Header (hex) : {header.hex()}")

    if header[:8] == b'\x89HDF\r\n\x1a\n':
        print("✅ Format HDF5 (.h5) détecté")
        return "h5"
    elif header[:4] == b'PK\x03\x04':
        print("📦 Format ZIP (.keras natif Keras 3) détecté")
        return "keras_zip"
    elif b'<html' in header.lower():
        print("❌ Page HTML d'erreur Google Drive")
        return "html_error"
    else:
        print("❓ Format inconnu")
        return "unknown"

    return size


def download_model():
    """
    Télécharge le modèle depuis Google Drive si non présent localement.
    Vérifie l'intégrité du fichier après téléchargement.
    """
    # Vérification si fichier déjà présent et valide
    if os.path.exists(MODEL_PATH):
        format_detecte = diagnose_model_file(MODEL_PATH)
        if format_detecte == "h5":
            print(f"✅ Modèle HDF5 valide déjà présent : {MODEL_PATH}")
            return MODEL_PATH
        else:
            print("⚠️ Fichier corrompu ou invalide détecté, suppression...")
            os.remove(MODEL_PATH)

    if not MODEL_DRIVE_URL:
        print("❌ Aucun lien Google Drive configuré.")
        return None

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"🚀 Téléchargement du modèle depuis Google Drive...")
    print(f"   URL : {MODEL_DRIVE_URL}")

    try:
        # fuzzy=True extrait l'ID même avec des paramètres complexes
        # use_cookies=False évite les problèmes de session expirée
        gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True, use_cookies=False)

        # Vérification post-téléchargement
        file_size = os.path.getsize(MODEL_PATH)
        print(f"📦 Fichier téléchargé : {file_size / 1024 / 1024:.2f} MB")

        if file_size < 1000000:  # < 1Mo = probablement page HTML
            with open(MODEL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)
                if '<html' in content.lower() or '<!doctype' in content.lower():
                    os.remove(MODEL_PATH)
                    raise ValueError(
                        "Google Drive a renvoyé une page HTML au lieu du modèle.\n"
                        "Cause probable : fichier > 100MB nécessite confirmation virus.\n"
                        "Solution : vérifiez que le fichier est bien en mode 'Partage public'"
                    )

        # Vérifie le format final
        format_final = diagnose_model_file(MODEL_PATH)
        if format_final != "h5":
            raise ValueError(f"Format de fichier non reconnu après téléchargement : {format_final}. Attendu : HDF5 (.h5)")

        print(f"✨ Modèle téléchargé et validé : {MODEL_PATH}")

    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return None

    return MODEL_PATH


def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    Construit le modèle avec Transfer Learning (MobileNetV2).
    Cette architecture doit être IDENTIQUE à celle utilisée pendant l'entraînement.
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_model():
    """
    Charge le modèle .h5 depuis le disque ou le télécharge depuis Google Drive.
    """
    if not os.path.exists(MODEL_PATH):
        path = download_model()
        if path is None:
            raise FileNotFoundError(f"Impossible de télécharger le modèle. Vérifiez MODEL_DRIVE_URL.")

    # Diagnostic avant chargement
    format_fichier = diagnose_model_file(MODEL_PATH)

    if format_fichier == "html_error":
        os.remove(MODEL_PATH)
        raise ValueError("Fichier corrompu (page HTML). Supprimé. Relancez pour retélécharger.")

    if format_fichier != "h5":
        raise ValueError(f"Format invalide : {format_fichier}. Le fichier doit être au format HDF5 (.h5)")

    # Chargement du modèle HDF5
    try:
        print(f"🔄 Chargement du modèle : {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # Recompile avec les paramètres locaux
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Modèle chargé avec succès")
        return model

    except Exception as e:
        print(f"❌ Erreur chargement : {e}")

        # Fallback : reconstruction + poids
        print("🔄 Tentative de reconstruction d'architecture + poids...")
        try:
            model = build_model()
            model.load_weights(MODEL_PATH)
            print("✅ Poids chargés sur architecture reconstruite")
            return model
        except Exception as e2:
            print(f"❌ Échec reconstruction : {e2}")
            raise RuntimeError(
                f"Impossible de charger le modèle.\n"
                f"Erreur chargement : {e}\n"
                f"Erreur reconstruction : {e2}"
            )
