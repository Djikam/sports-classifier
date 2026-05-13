#!/usr/bin/env python3
"""
convert_model.py
================
Convertit un modèle .keras (format ZIP Keras 3) en .h5 (HDF5)
Pour exécution dans VS Code / terminal local.

Usage:
    python convert_model.py

Le script va :
1. Charger ton best_model.keras
2. Le convertir en best_model.h5
3. Vérifier que le .h5 est valide
4. Afficher les instructions pour upload sur Google Drive
"""

import os
import sys
import gc
import hashlib

# ─── CONFIGURATION : MODIFIE CES 2 CHEMINS ─────────────────
# Chemin vers ton modèle .keras actuel (modifie selon ton projet)
INPUT_PATH = "models/best_model.keras"          # ← MODIFIE ICI si besoin

# Où sauvegarder le .h5 converti (même dossier par défaut)
OUTPUT_DIR = os.path.dirname(INPUT_PATH) or "."
OUTPUT_NAME = "best_model.h5"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
# ───────────────────────────────────────────────────────────


def print_section(title):
    """Affiche une section formatée."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_file_signature(path):
    """Vérifie la signature binaire du fichier."""
    with open(path, 'rb') as f:
        header = f.read(8)

    if header[:8] == b'\x89HDF\r\n\x1a\n':
        return "hdf5", "✅ Fichier HDF5 valide"
    elif header[:4] == b'PK\x03\x04':
        return "zip", "📦 Fichier ZIP (format .keras natif Keras 3)"
    elif b'<html' in header.lower():
        return "html", "❌ Page HTML (fichier corrompu ou erreur téléchargement)"
    else:
        return "unknown", f"⚠️ Format inconnu (header: {header.hex()})"


def get_file_hash(path):
    """Calcule le hash MD5 du fichier."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    print("\n🚀 CONVERTISSEUR .keras → .h5")
    print("   TensorFlow/Keras Model Format Converter")

    # ─── ÉTAPE 1 : VÉRIFICATION SOURCE ─────────────────────
    print_section("1. VÉRIFICATION DU FICHIER SOURCE")

    print(f"📁 Chemin source : {os.path.abspath(INPUT_PATH)}")

    if not os.path.exists(INPUT_PATH):
        print(f"\n❌ ERREUR : Fichier non trouvé !")
        print(f"   {INPUT_PATH}")
        print(f"\n💡 Vérifie que le fichier existe et que le chemin est correct.")
        print(f"   Tu peux modifier INPUT_PATH en haut du script.")
        sys.exit(1)

    file_size = os.path.getsize(INPUT_PATH)
    print(f"📊 Taille : {file_size / 1024 / 1024:.2f} MB ({file_size:,} bytes)")

    fmt, msg = check_file_signature(INPUT_PATH)
    print(f"{msg}")

    if fmt == "html":
        print("\n❌ Le fichier est une page HTML, pas un modèle Keras.")
        print("   Cela arrive quand Google Drive renvoie une erreur au lieu du fichier.")
        sys.exit(1)

    # ─── ÉTAPE 2 : IMPORT TENSORFLOW ───────────────────────
    print_section("2. IMPORT DE TENSORFLOW")

    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} importé")

        # Vérifie si GPU disponible (optionnel)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 GPU détecté : {len(gpus)} device(s)")
        else:
            print("💻 Mode CPU (pas de GPU détecté)")

    except ImportError:
        print("❌ TensorFlow n'est pas installé !")
        print("\n💡 Installe-le avec :")
        print("   pip install tensorflow")
        sys.exit(1)

    # ─── ÉTAPE 3 : CHARGEMENT DU MODÈLE ────────────────────
    print_section("3. CHARGEMENT DU MODÈLE .keras")

    try:
        print("⏳ Chargement en cours... (peut prendre quelques secondes)")
        model = tf.keras.models.load_model(INPUT_PATH)
        print("✅ Modèle chargé avec succès !")

        # Affiche quelques infos
        print(f"\n📐 Architecture :")
        print(f"   - Nombre de couches : {len(model.layers)}")
        print(f"   - Entrée : {model.input_shape}")
        print(f"   - Sortie : {model.output_shape}")
        print(f"   - Paramètres totaux : {model.count_params():,}")

    except Exception as e:
        print(f"\n❌ Erreur lors du chargement : {e}")
        print(f"\n💡 Causes possibles :")
        print(f"   - Le fichier .keras est corrompu (retélécharge-le)")
        print(f"   - Version TensorFlow incompatible avec le format .keras")
        print(f"   - Dépendances manquantes (custom objects)")
        sys.exit(1)

    # ─── ÉTAPE 4 : CONVERSION .h5 ──────────────────────────
    print_section("4. CONVERSION EN .h5 (HDF5)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Dossier de sortie : {os.path.abspath(OUTPUT_DIR)}")

    try:
        print(f"⏳ Sauvegarde en cours...")
        model.save(OUTPUT_PATH, save_format='h5')
        print(f"✅ Fichier créé : {OUTPUT_PATH}")

        h5_size = os.path.getsize(OUTPUT_PATH)
        print(f"📊 Taille .h5 : {h5_size / 1024 / 1024:.2f} MB ({h5_size:,} bytes)")

        # Vérification signature
        fmt_h5, msg_h5 = check_file_signature(OUTPUT_PATH)
        print(f"{msg_h5}")

        if fmt_h5 != "hdf5":
            print("\n⚠️  Le fichier .h5 n'a pas la signature HDF5 attendue !")

    except Exception as e:
        print(f"\n❌ Erreur sauvegarde : {e}")
        sys.exit(1)

    # ─── ÉTAPE 5 : VÉRIFICATION ────────────────────────────
    print_section("5. VÉRIFICATION DU FICHIER .h5")

    try:
        print("⏳ Rechargement du .h5 pour vérification...")
        model_test = tf.keras.models.load_model(OUTPUT_PATH)
        print("✅ Le fichier .h5 se charge correctement !")

        # Compare les architectures
        if model_test.count_params() == model.count_params():
            print("✅ Nombre de paramètres identique")
        else:
            print("⚠️  Nombre de paramètres différent !")

        # Compare un forward pass simple
        import numpy as np
        dummy_input = np.zeros((1, *model.input_shape[1:]))
        pred_original = model.predict(dummy_input, verbose=0)
        pred_test = model_test.predict(dummy_input, verbose=0)

        if np.allclose(pred_original, pred_test):
            print("✅ Prédictions identiques (forward pass validé)")
        else:
            print("⚠️  Prédictions différentes !")

    except Exception as e:
        print(f"⚠️  Erreur vérification : {e}")

    # ─── ÉTAPE 6 : INFOS FICHIER ───────────────────────────
    print_section("6. INFOS DU FICHIER CONVERTI")

    md5_hash = get_file_hash(OUTPUT_PATH)
    print(f"📄 Nom : {OUTPUT_NAME}")
    print(f"📍 Chemin complet : {os.path.abspath(OUTPUT_PATH)}")
    print(f"📊 Taille : {h5_size / 1024 / 1024:.2f} MB")
    print(f"🔑 MD5 : {md5_hash}")

    # ─── ÉTAPE 7 : INSTRUCTIONS UPLOAD ─────────────────────
    print_section("7. PROCHAINES ÉTAPES : UPLOAD SUR GOOGLE DRIVE")

    print("""
📋 POUR METTRE LE FICHIER SUR GOOGLE DRIVE :

   1. Va sur https://drive.google.com
   2. Upload le fichier : best_model.h5
      (glisse-dépose ou clic droit → Upload)
   3. Clic droit sur le fichier → "Partager"
   4. Règle sur "Tous ceux qui ont le lien" (mode "Lecteur")
   5. Copie le lien (il ressemble à ça) :
      https://drive.google.com/file/d/1xG2mIfFIhg607MdF-lTGIMif14L3fKK8/view
                                    └──────────────────────────────┘
                                              ↑ COPIE CETTE PARTIE

📋 DANS TON CODE src/model.py, REMPLACE :

   MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1xG2mIfFIhg607MdF-lTGIMif14L3fKK8&export=download&confirm=t"
   MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")

   (remplace 1xG2mIfFIhg607MdF-lTGIMif14L3fKK8 par TON ID)

⚠️  IMPORTANT :
   - Garde le fichier .keras original en backup !
   - Le .h5 est plus compatible avec les vieilles versions TF
   - Si le fichier fait > 100MB, Google Drive peut demander une
     confirmation de téléchargement (gdown gère ça avec fuzzy=True)
""")

    # ─── NETTOYAGE ─────────────────────────────────────────
    print_section("8. NETTOYAGE")

    del model
    try:
        del model_test
    except NameError:
        pass
    gc.collect()
    print("✅ Mémoire libérée")

    print("\n" + "="*60)
    print("  🎉 CONVERSION TERMINÉE AVEC SUCCÈS")
    print("="*60)
    print(f"\n📁 Fichier converti : {os.path.abspath(OUTPUT_PATH)}")
    print("\n👉 Copie ce fichier sur Google Drive et mets à jour l'URL")
    print("   dans src/model.py pour que ton Streamlit fonctionne !")


if __name__ == "__main__":
    main()
