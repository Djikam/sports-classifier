import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.set_page_config(page_title="Analyse", page_icon="📊")

st.title("📊 Analyse et Visualisation")

st.markdown("""
Cette page permet d'analyser les performances du modèle et de visualiser des exemples.
""")

# Section 1 : Distribution des classes
st.header("1. Distribution des classes")
st.info("📌 *Cette section nécessite le dataset local. Exécutez le notebook d'exploration pour générer les graphiques.*")

# Placeholder pour graphique
st.markdown("""
**Attendu (après exécution du notebook) :**
- Histogramme de distribution des 100 classes
- Vérification du déséquilibre éventuel
- Nombre d'images par classe (~130 par classe en moyenne)
""")

# Section 2 : Matrice de confusion
st.header("2. Matrice de confusion")
st.info("📌 *Générée après entraînement du modèle.*")

st.markdown("""
La matrice de confusion permet d'identifier :
- Les sports les mieux reconnus (diagonale forte)
- Les confusions fréquentes (hors diagonale)
- Exemples de confusions typiques :
  - 🏈 Football américain ↔ 🏉 Rugby
  - 🎾 Tennis ↔ 🏸 Badminton
  - 🏊 Natation ↔ 🤽 Water-polo
""")

# Section 3 : Courbes d'entraînement
st.header("3. Courbes d'entraînement")
st.info("📌 *Historique d'entraînement sauvegardé après le notebook 02_training.*")

st.markdown("""
**Métriques attendues :**
- **Loss** : diminution rapide puis plateau (~0.3-0.5)
- **Accuracy** : croissance rapide vers ~93%
- **Validation** : suivi pour détecter l'overfitting
- **Epochs optimales** : généralement 10-20 avec early stopping
""")

# Section 4 : Exemples de prédictions
st.header("4. Exemples de prédictions")

uploaded_files = st.file_uploader(
    "Upload plusieurs images pour comparaison",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    from src.inference import SportsClassifier

    try:
        classifier = SportsClassifier()

        cols = st.columns(min(len(uploaded_files), 3))

        for idx, uploaded_file in enumerate(uploaded_files):
            col = cols[idx % 3]
            with col:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)

                cls, prob = classifier.predict_class(image)
                st.success(f"**{cls}** ({prob:.1%})")
    except Exception as e:
        st.error(f"Erreur : {e}")
