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
page = st.sidebar.radio("", ["Classification", "À propos"], label_visibility="collapsed")


# Chargement du modèle (mis en cache)
@st.cache_resource(show_spinner=False)
def load_classifier():
    with st.spinner("Chargement du modèle... (peut prendre 30-60s au premier démarrage)"):
        return SportsClassifier()


# Gestion robuste du chargement
classifier = None
model_loaded = False
load_error = None

try:
    classifier = load_classifier()
    model_loaded = True
except Exception as e:
    load_error = str(e)
    st.error(f"❌ Erreur de chargement du modèle : {e}")

    if "keras_zip" in load_error.lower():
        st.markdown("""
        ### 🔧 Format .keras détecté
        Le fichier est au format ZIP natif Keras 3. 
        Le chargement des poids sur architecture reconstruite a échoué.

        **Solution :** Convertissez le modèle en .h5 :
        ```python
        model = tf.keras.models.load_model("best_model.keras")
        model.save("best_model.h5")
        ```
        """)
    elif "download" in load_error.lower() or "url" in load_error.lower():
        st.markdown("""
        ### 🔧 Problème de téléchargement
        Vérifiez l'URL du modèle dans les variables d'environnement Render.
        """)
    else:
        st.markdown("""
        ### 🔧 Erreur de chargement
        Vérifiez les logs pour plus de détails.
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
            st.image(image, use_column_width=True)
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
