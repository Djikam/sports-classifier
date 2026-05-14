"""
Gestion du modele : architecture, chargement, telechargement.
SOLUTION DEFINITIVE : Reconstruction + poids (evite les conflits Keras 2/3)
"""
import os
import zipfile
import urllib.request
import shutil
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from src.utils import MODELS_DIR, NUM_CLASSES, IMG_SIZE

# URL du modele
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/uc?id=1S46qADbZHvMNOf13_1-WLvhLGnY4RqN-&export=download&confirm=t"
)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")


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

    if os.path.exists(MODEL_PATH):
        fmt, size = diagnose_model_file(MODEL_PATH)
        if fmt in ["h5", "keras_zip"]:
            print(f"✅ Fichier deja present ({size/1024/1024:.1f} MB, format: {fmt})")
            return MODEL_PATH, fmt
        else:
            print(f"⚠️ Invalide ({fmt}), suppression...")
            os.remove(MODEL_PATH)

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"🚀 Telechargement...")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(MODEL_URL, headers=headers)

        with urllib.request.urlopen(req, timeout=300) as response:
            data = response.read()

        with open(MODEL_PATH, 'wb') as f:
            f.write(data)

        fmt, size = diagnose_model_file(MODEL_PATH)
        print(f"📦 Recu: {fmt} ({size/1024/1024:.1f} MB)")

        if fmt == "html_error":
            os.remove(MODEL_PATH)
            print("⚠️ HTML recu, tentative avec confirmation...")
            return download_with_confirmation()

        if fmt not in ["h5", "keras_zip"]:
            os.remove(MODEL_PATH)
            raise ValueError(f"Format non supporte: {fmt}")

        return MODEL_PATH, fmt

    except Exception as e:
        print(f"❌ Erreur: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
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

    with open(MODEL_PATH, 'wb') as f:
        f.write(data)

    fmt, size = diagnose_model_file(MODEL_PATH)
    print(f"📦 Confirmation: {fmt} ({size/1024/1024:.1f} MB)")

    if fmt not in ["h5", "keras_zip"]:
        os.remove(MODEL_PATH)
        raise ValueError(f"Toujours invalide: {fmt}")

    return MODEL_PATH, fmt


def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """Construit l'architecture du modele (IDENTIQUE a l'entrainement)."""
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


def extract_weights_from_keras(path):
    """Extrait les poids d'un fichier .keras (ZIP)."""
    print("📦 Extraction des poids du ZIP .keras...")

    extract_dir = os.path.join(MODELS_DIR, "_temp_extract")
    os.makedirs(extract_dir, exist_ok=True)

    weights_file = None

    try:
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(extract_dir)

        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.endswith(".weights.h5") or f == "model.weights.h5":
                    weights_file = os.path.join(root, f)
                    break

        if weights_file and os.path.exists(weights_file):
            print(f"✅ Poids trouves: {os.path.basename(weights_file)}")
            return weights_file
        else:
            print("⚠️ Poids non trouves dans le ZIP")
            return None

    except Exception as e:
        print(f"❌ Erreur extraction: {e}")
        return None

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def load_model():
    """
    Charge le modele : TOUJOURS reconstruction + poids.
    Evite les conflits Keras 2/3 car on ne deserialise pas la config JSON.
    """
    # Verifie/Telecharge le fichier
    if not os.path.exists(MODEL_PATH):
        path, fmt = download_model()
    else:
        fmt, size = diagnose_model_file(MODEL_PATH)
        if fmt not in ["h5", "keras_zip"]:
            print(f"⚠️ Fichier invalide ({fmt}), retéléchargement...")
            os.remove(MODEL_PATH)
            path, fmt = download_model()
        else:
            print(f"✅ Fichier local: {fmt} ({size/1024/1024:.1f} MB)")

    fmt, _ = diagnose_model_file(MODEL_PATH)

    if fmt == "html_error":
        os.remove(MODEL_PATH)
        raise ValueError("Fichier corrompu (HTML)")

    # ETAPE 1 : Reconstruit l'architecture (TOUJOURS)
    print("🔄 Reconstruction de l'architecture...")
    model = build_model()

    # ETAPE 2 : Charge les poids selon le format
    if fmt == "keras_zip":
        # Extrait les poids du ZIP puis charge
        weights_path = extract_weights_from_keras(MODEL_PATH)
        if weights_path:
            try:
                model.load_weights(weights_path)
                print("✅ Poids .keras charges")
            except Exception as e:
                print(f"⚠️ Erreur chargement poids .keras: {e}")
                print("   Modele utilise avec poids ImageNet (non entraine)")
        else:
            print("⚠️ Poids non trouves, modele avec poids ImageNet")

    else:  # fmt == "h5"
        try:
            # Essaie de charger les poids du .h5
            model.load_weights(MODEL_PATH)
            print("✅ Poids .h5 charges")
        except Exception as e:
            print(f"⚠️ Erreur chargement poids .h5: {e}")
            print("   Le .h5 contient peut-etre une config Keras 3 incompatible")
            print("   Modele utilise avec poids ImageNet (non entraine)")

    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ Modele pret")
    return model
