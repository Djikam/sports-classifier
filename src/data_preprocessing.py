"""
Préparation des données : chargement, augmentation, preprocessing.
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.utils import DATA_DIR, IMG_SIZE, BATCH_SIZE


def get_data_generators(validation_split=0.0):
    """
    Crée les générateurs de données pour l'entraînement.

    Returns:
        train_generator, valid_generator, test_generator
    """
    # Chemins
    train_dir = os.path.join(DATA_DIR, "train")
    valid_dir = os.path.join(DATA_DIR, "valid")
    test_dir = os.path.join(DATA_DIR, "test")

    # Data augmentation pour le training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )

    # Pas d'augmentation pour valid/test
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Générateurs
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, valid_generator, test_generator


def get_class_indices():
    """
    Retourne le mapping index -> nom de classe.
    """
    train_dir = os.path.join(DATA_DIR, "train")
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode='categorical'
    )
    # Inverser le dictionnaire
    class_indices = {v: k for k, v in generator.class_indices.items()}
    return class_indices
