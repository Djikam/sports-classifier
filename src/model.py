"""
Gestion du modele : architecture, chargement, telechargement.
Accepte les deux formats : .h5 (HDF5) et .keras (ZIP Keras 3)
"""
import os
import zipfile
import json
import urllib.request
import shutil
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from src.utils import MODELS_DIR, NUM_CLASSES, IMG_SIZE

# URL du modele (Google Drive ou GitHub Releases)
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/uc?id=1S46qADbZHvMNOf13_1-WLvhLGnY4RqN-&export=download&confirm=t"
)

# Chemins possibles
MODEL_PATH_H5 = os.path.join(MODELS_DIR, "best_model.h5")
MODEL_PATH_KERAS = os.path.join(MODELS_DIR, "best_model.keras")


def diagnose_model_file(path):
    """Diagnostic du fichier modele."""
    if not os.path.exists(path):
        return None, 0

    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        header = f.read(8)

    if header[:8] == b'\x89HDF\r\n\x1a\n':
        return "h5", size
    elif header[:4] == b'PK\x03\x04':
        return "keras_zip", size
    elif b'<html' in header.lower():
        return "html_error", size
    else:
        return "unknown", size


def download_model():
    """Telecharge le modele depuis l'URL configuree."""

    print(f"🔧 URL: {MODEL_URL[:60]}...")

    # Verifie si un fichier valide existe deja
    for path, expected in [(MODEL_PATH_H5, "h5"), (MODEL_PATH_KERAS, "keras_zip")]:
        if os.path.exists(path):
            fmt, size = diagnose_model_file(path)
            if fmt == expected:
                print(f"✅ {expected.upper()} deja present ({size/1024/1024:.1f} MB)")
                return path, fmt
            else:
                print(f"⚠️ {path} invalide ({fmt}), suppression...")
                os.remove(path)

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"🚀 Telechargement...")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(MODEL_URL, headers=headers)

        with urllib.request.urlopen(req, timeout=300) as response:
            data = response.read()

        # Sauvegarde temporaire pour diagnostic
        temp_path = os.path.join(MODELS_DIR, "best_model_temp")
        with open(temp_path, 'wb') as f:
            f.write(data)

        fmt, size = diagnose_model_file(temp_path)
        print(f"📦 Recu: {fmt} ({size/1024/1024:.1f} MB)")

        if fmt == "html_error":
            os.remove(temp_path)
            print("⚠️ HTML recu, tentative avec confirmation...")
            return download_with_confirmation()

        if fmt == "h5":
            os.rename(temp_path, MODEL_PATH_H5)
            return MODEL_PATH_H5, "h5"
        elif fmt == "keras_zip":
            os.rename(temp_path, MODEL_PATH_KERAS)
            return MODEL_PATH_KERAS, "keras_zip"
        else:
            os.remove(temp_path)
            raise ValueError(f"Format non supporte: {fmt}")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise


def download_with_confirmation():
    """Gere la confirmation Google Drive."""
    import re

    match = re.search(r'id=([^&]+)', MODEL_URL)
    if not match:
        raise ValueError("ID non trouve")

    file_id = match.group(1)
    confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"

    print(f"🔄 Confirmation: {confirm_url[:60]}...")

    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(confirm_url, headers=headers)

    with urllib.request.urlopen(req, timeout=300) as response:
        data = response.read()

    temp_path = os.path.join(MODELS_DIR, "best_model_temp")
    with open(temp_path, 'wb') as f:
        f.write(data)

    fmt, size = diagnose_model_file(temp_path)
    print(f"📦 Confirmation: {fmt} ({size/1024/1024:.1f} MB)")

    if fmt == "h5":
        os.rename(temp_path, MODEL_PATH_H5)
        return MODEL_PATH_H5, "h5"
    elif fmt == "keras_zip":
        os.rename(temp_path, MODEL_PATH_KERAS)
        return MODEL_PATH_KERAS, "keras_zip"
    else:
        os.remove(temp_path)
        raise ValueError(f"Toujours invalide: {fmt}")


def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """Construit le modele avec Transfer Learning."""
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


def load_model_from_keras_zip(path):
    """Charge un modele .keras (ZIP) en reconstruisant l'architecture + poids."""
    print("📦 Extraction du ZIP .keras...")

    extract_dir = os.path.join(MODELS_DIR, "_temp_extract")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(extract_dir)

        # Cherche config et poids
        config_file = None
        weights_file = None

        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                full = os.path.join(root, f)
                if f == "config.json":
                    config_file = full
                elif f.endswith(".weights.h5") or f == "model.weights.h5":
                    weights_file = full

        print(f"📄 Config: {config_file}")
        print(f"⚖️  Poids: {weights_file}")

        # Reconstruit l'architecture
        model = build_model()

        if weights_file and os.path.exists(weights_file):
            model.load_weights(weights_file)
            print("✅ Poids charges")
        else:
            print("⚠️ Poids non trouves, modele vierge")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def load_model():
    """Charge le modele, gere .h5 et .keras"""

    # Cherche un fichier existant
    path = None
    fmt = None

    if os.path.exists(MODEL_PATH_H5):
        fmt, _ = diagnose_model_file(MODEL_PATH_H5)
        if fmt == "h5":
            path = MODEL_PATH_H5
    elif os.path.exists(MODEL_PATH_KERAS):
        fmt, _ = diagnose_model_file(MODEL_PATH_KERAS)
        if fmt == "keras_zip":
            path = MODEL_PATH_KERAS

    # Si aucun fichier valide, telecharge
    if path is None:
        path, fmt = download_model()

    print(f"\n📁 Fichier: {path}")
    print(f"📦 Format: {fmt}")

    # Chargement selon le format
    if fmt == "h5":
        print("🔄 Chargement HDF5...")
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Modele .h5 charge")
        return model

    elif fmt == "keras_zip":
        print("🔄 Chargement .keras (ZIP)...")
        return load_model_from_keras_zip(path)

    else:
        raise ValueError(f"Format non gere: {fmt}")
