"""
Gestion du modèle : architecture, chargement, téléchargement.
"""
import os
import urllib.request
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

from src.utils import MODELS_DIR, NUM_CLASSES, IMG_SIZE

# URL Google Drive - fichier .h5
MODEL_DRIVE_URL = os.environ.get(
    "MODEL_DRIVE_URL",
    "https://drive.google.com/uc?id=1S46qADbZHvMNOf13_1-WLvhLGnY4RqN-&export=download&confirm=t"
)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")


def diagnose_model_file(path):
    """Diagnostic du fichier modèle."""
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
    """Télécharge le modèle depuis Google Drive avec gestion des confirmations."""
    
    print(f"🔧 URL configurée: {MODEL_DRIVE_URL[:50]}...")
    
    # Vérifie si déjà présent
    if os.path.exists(MODEL_PATH):
        fmt, size = diagnose_model_file(MODEL_PATH)
        if fmt == "h5":
            print(f"✅ Modèle déjà présent ({size/1024/1024:.1f} MB)")
            return MODEL_PATH
        else:
            print(f"⚠️ Fichier invalide ({fmt}), suppression...")
            os.remove(MODEL_PATH)

    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"🚀 Téléchargement depuis Google Drive...")

    try:
        # Tentative 1 : URL directe
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(MODEL_DRIVE_URL, headers=headers)
        
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
        
        with open(MODEL_PATH, 'wb') as f:
            f.write(data)
        
        fmt, size = diagnose_model_file(MODEL_PATH)
        print(f"📦 Fichier reçu: {fmt} ({size/1024/1024:.1f} MB)")
        
        # Si c'est du HTML, c'est une page de confirmation
        if fmt == "html_error":
            print("⚠️ Confirmation Google Drive requise...")
            os.remove(MODEL_PATH)
            return download_with_confirmation()
        
        if fmt != "h5":
            os.remove(MODEL_PATH)
            raise ValueError(f"Format invalide: {fmt}")
        
        print("✅ Téléchargement réussi")
        return MODEL_PATH
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise


def download_with_confirmation():
    """Gère la confirmation de téléchargement pour les gros fichiers Google Drive."""
    import re
    
    # Extrait l'ID
    match = re.search(r'id=([^&]+)', MODEL_DRIVE_URL)
    if not match:
        raise ValueError("ID Google Drive non trouvé dans l'URL")
    
    file_id = match.group(1)
    confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    
    print(f"🔄 Tentative avec confirmation: {confirm_url[:60]}...")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(confirm_url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
    
    with open(MODEL_PATH, 'wb') as f:
        f.write(data)
    
    fmt, size = diagnose_model_file(MODEL_PATH)
    print(f"📦 Résultat: {fmt} ({size/1024/1024:.1f} MB)")
    
    if fmt != "h5":
        os.remove(MODEL_PATH)
        raise ValueError(f"Toujours format invalide: {fmt}")
    
    print("✅ Téléchargement confirmé réussi")
    return MODEL_PATH


def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """Construit le modèle avec Transfer Learning."""
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
    """Charge le modèle .h5 depuis le disque ou le télécharge."""
    if not os.path.exists(MODEL_PATH):
        download_model()
    
    fmt, size = diagnose_model_file(MODEL_PATH)
    
    if fmt == "html_error":
        os.remove(MODEL_PATH)
        raise ValueError("Fichier corrompu (HTML)")
    
    if fmt != "h5":
        raise ValueError(f"Format invalide: {fmt}. Attendu: HDF5 (.h5)")
    
    print(f"🔄 Chargement du modèle ({size/1024/1024:.1f} MB)...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Modèle chargé avec succès")
    return model