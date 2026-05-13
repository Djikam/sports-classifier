import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import streamlit as st
import sys

# Ajouter le répertoire parent au path pour importer src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import numpy as np
from src.inference import SportsClassifier
from app.utils import plot_prediction_probs, display_metrics

# Configuration de la page
st.set_page_config(
    page_title="Classificateur de Sports",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high { color: #2ecc71; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏅 Navigation")
page = st.sidebar.radio("", ["Classification", "À propos"])


# Chargement du modèle (mis en cache)
@st.cache_resource(show_spinner=False)
def load_classifier():
    with st.spinner("Chargement du modèle... (peut prendre 30-60s au premier démarrage)"):
        return SportsClassifier()


# Gestion robuste du chargement avec affichage d'aide en cas d'erreur
classifier = None
model_loaded = False
load_error = None

try:
    classifier = load_classifier()
    model_loaded = True
except Exception as e:
    load_error = str(e)
    st.error(f"❌ Erreur de chargement du modèle : {e}")

    # Affiche un guide de dépannage contextuel
    if "html" in load_error.lower() or "drive" in load_error.lower() or "download" in load_error.lower():
        st.markdown("""
        ### 🔧 Problème de téléchargement Google Drive

        Le modèle n'a pas pu être téléchargé depuis Google Drive.

        **Solutions :**
        - Vérifiez que le fichier `best_model.h5` est bien sur Drive
        - Vérifiez que le lien est en mode **"Tous ceux qui ont le lien"** (Partager → Accès général)
        - Essayez de mettre à jour `MODEL_DRIVE_URL` dans `src/model.py`
        - Alternative : hébergez sur [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
        """)
    elif "hdf5" in load_error.lower() or "signature" in load_error.lower():
        st.markdown("""
        ### 🔧 Problème de format de fichier

        Le fichier téléchargé n'est pas un modèle HDF5 (.h5) valide.

        **Solutions :**
        - Vérifiez que le fichier sur Drive est bien `best_model.h5` (pas .keras)
        - Supprimez le fichier local corrompu : supprimez `models/best_model.h5`
        - Relancez l'application pour retélécharger
        """)
    else:
        st.markdown("""
        ### 🔧 Erreur de chargement du modèle

        **Solutions :**
        - Vérifiez que TensorFlow est installé : `pip install tensorflow`
        - Vérifiez les logs dans le terminal
        - Essayez de reconstruire le modèle avec `build_model()` dans `src/model.py`
        """)


# Page Classification
if page == "Classification":
    st.markdown("<p class='main-header'>🏆 Classificateur d'Images de Sports</p>", unsafe_allow_html=True)
    st.write("**Upload une photo d'un sport et le modèle prédit la discipline parmi 100 sports !**")

    uploaded_file = st.file_uploader(
        "📤 Choisis une image (JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        help="L'image sera redimensionnée en 224×224 pour le modèle"
    )

    if uploaded_file is not None and model_loaded:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Image uploadée")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.info(f"**Dimensions** : {image.size[0]}×{image.size[1]} pixels | **Mode** : {image.mode}")

        with col2:
            st.subheader("🎯 Prédiction")
            with st.spinner("Analyse en cours..."):
                results = classifier.predict(image, top_k=5)
                top_class, top_prob = results[0]

                confidence_class = "confidence-high" if top_prob > 0.7 else "confidence-medium" if top_prob > 0.4 else "confidence-low"

                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Sport prédit : <span class="{confidence_class}">{top_class}</span></h3>
                    <p style="font-size: 1.2rem;">Confiance : <span class="{confidence_class}">{top_prob:.2%}</span></p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📊 Top 5 prédictions")
                fig = plot_prediction_probs(results)
                st.pyplot(fig)

                st.subheader("📋 Détails")
                for i, (cls, prob) in enumerate(results, 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    bar = "█" * int(prob * 20)
                    st.write(f"{emoji} **{cls}** : {prob:.2%} {bar}")

    elif uploaded_file is not None and not model_loaded:
        st.warning("⚠️ Le modèle n'est pas chargé. Vérifiez les logs d'erreur ci-dessus.")

    else:
        st.info("👆 Upload une image pour commencer la classification !")
        st.markdown("""
        ### 💡 Conseils pour de meilleurs résultats :
        - Utilisez une image **claire et bien cadrée**
        - Le sport doit être **visible et identifiable**
        - Formats acceptés : JPG, PNG, JPEG
        - L'image sera automatiquement redimensionnée en **224×224**
        """)

# Page À propos
elif page == "À propos":
    st.markdown('<p class="main-header">ℹ️ À propos du projet</p>', unsafe_allow_html=True)
    st.markdown("""
    ### 🧠 Architecture du modèle
    - **Base** : MobileNetV2 (pré-entraîné sur ImageNet)
    - **Couches ajoutées** : GlobalAveragePooling2D + Dense(512) + Dropout(0.3) + Dense(100)
    - **Transfer Learning** : Base frozen + fine-tuning optionnel
    - **Accuracy** : ~93% (validation) / ~91% (test)

    ### 📊 Dataset
    - **Source** : [Kaggle - 100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
    - **100 classes** de sports variés
    - **~13 000 images** au total (train/valid/test)
    - Images déjà redimensionnées en **224×224**

    ### 🛠️ Stack Technique
    | Composant | Technologie |
    |-----------|-------------|
    | Deep Learning | TensorFlow/Keras |
    | Transfer Learning | MobileNetV2 |
    | App Web | Streamlit |
    | Déploiement | Render |
    """)

# Métriques et footer
display_metrics()
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 - Sports Classifier v1.0")
