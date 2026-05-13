#!/usr/bin/env python3
"""
Script d'entraînement standalone.
Usage: python src/train.py
"""
import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model
from src.data_preprocessing import get_data_generators
from src.utils import MODELS_DIR


def train(epochs_phase1=15, epochs_phase2=10):
    """Pipeline complet d'entraînement."""
    print("=" * 50)
    print("🏋️  ENTRAÎNEMENT SPORTS CLASSIFIER")
    print("=" * 50)

    # 1. Données
    print("\n📂 Chargement des données...")
    train_gen, valid_gen, test_gen = get_data_generators()
    print(f"   Train: {train_gen.samples} | Valid: {valid_gen.samples} | Test: {test_gen.samples}")

    # 2. Modèle
    print("\n🧠 Construction du modèle (MobileNetV2)...")
    model = build_model(num_classes=100, img_size=224)
    print(f"   Paramètres totaux: {model.count_params():,}")

    # 3. Callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # 4. Phase 1
    print(f"\n🔥 Phase 1: Entraînement (base frozen) - {epochs_phase1} epochs")
    history1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs_phase1,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Fine-tuning
    print(f"\n🔧 Phase 2: Fine-tuning (défreeze partiel) - {epochs_phase2} epochs")
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs_phase2,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Évaluation
    print("\n📊 Évaluation finale sur test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"   ✅ Test Accuracy: {test_acc:.4f}")
    print(f"   ✅ Test Loss: {test_loss:.4f}")

    # 7. Sauvegarde
    final_path = os.path.join(MODELS_DIR, 'final_model.keras')
    model.save(final_path)
    print(f"\n💾 Modèle sauvegardé: {final_path}")
    print("\n🎉 Entraînement terminé !")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Sports Classifier")
    parser.add_argument("--epochs1", type=int, default=15, help="Epochs phase 1")
    parser.add_argument("--epochs2", type=int, default=10, help="Epochs phase 2")
    args = parser.parse_args()

    train(epochs_phase1=args.epochs1, epochs_phase2=args.epochs2)
