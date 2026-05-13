# 🏅 Sports Image Classifier

Classification d'images de sports à l'aide de Transfer Learning (MobileNetV2) et déploiement via Streamlit + Docker + Render.

## 📊 Dataset
- **Source** : [100 Sports Image Classification - Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- **Classes** : 100 sports différents
- **Images** : 224×224 pixels, split train/valid/test inclus
- **Taille** : ~445 MB

## 🎯 Objectif
Construire un modèle de Deep Learning capable de classifier une image de sport parmi 100 disciplines, avec une accuracy >90%.

## 🛠️ Technologies
- **Deep Learning** : TensorFlow/Keras + MobileNetV2 (Transfer Learning)
- **App Web** : Streamlit
- **Conteneurisation** : Docker
- **Déploiement** : Render (Web Service gratuit)
- **Data Science** : NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

## 📁 Structure du Projet

```
sports-classifier/
├── data/sports/              # Dataset (non versionné)
├── notebooks/
│   └── 01_exploration.ipynb  # EDA + Visualisation
│   └── 02_training.ipynb     # Entraînement du modèle
├── src/
│   ├── data_preprocessing.py # Chargement et preprocessing
│   ├── model.py              # Architecture + chargement
│   ├── inference.py          # Prédiction
│   └── utils.py              # Fonctions utilitaires
├── app/
│   ├── streamlit_app.py      # Application principale
│   └── pages/
│       └── 1_Analyse.py     # Page analyse
├── models/
│   └── best_model.keras      # Modèle entraîné (non versionné)
├── tests/                    # Tests unitaires
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation Locale

### 1. Cloner le repo
```bash
git clone https://github.com/TON_USER/sports-classifier.git
cd sports-classifier
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate   # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Télécharger le dataset
- Télécharger sur Kaggle : https://www.kaggle.com/datasets/gpiosenka/sports-classification
- Extraire dans `data/sports/`

### 5. Entraîner le modèle
Ouvrir et exécuter `notebooks/02_training.ipynb` (ou utiliser le script d'entraînement).

### 6. Lancer l'application Streamlit
```bash
streamlit run app/streamlit_app.py
```
L'app est accessible sur http://localhost:8501

## 🐳 Utilisation avec Docker

### Build
```bash
docker build -t sports-classifier .
```

### Run
```bash
docker run -p 8501:8501 sports-classifier
```
Accéder à http://localhost:8501

## 🌐 Déploiement sur Render

1. Créer un compte sur [Render.com](https://render.com)
2. New Web Service → Connecter le repo GitHub
3. Runtime : Docker (détection automatique du Dockerfile)
4. Déployer → Attendre le build (~5-10 min)
5. Copier l'URL dans `model_link.txt`

### ⚠️ Gestion du modèle trop lourd
GitHub limite les fichiers à 100MB. Le modèle `.keras` dépasse souvent cette taille.
**Solution utilisée** : Le modèle est stocké sur Google Drive et téléchargé automatiquement au démarrage de l'application via `gdown` (voir `src/model.py`).

## 📈 Résultats
- **Modèle** : MobileNetV2 + couches fully connected
- **Accuracy validation** : ~93%
- **Accuracy test** : ~91%
- **Temps d'entraînement** : ~15 min (GPU) / ~2h (CPU)

## 🧪 Tests
```bash
pytest tests/
```

## 📝 Notes
- Utilisation de `tensorflow-cpu` pour alléger le build Render gratuit
- Images redimensionnées en 224×224 pour MobileNetV2
- Normalisation des pixels entre 0 et 1

## 👤 Auteur
Projet réalisé dans le cadre d'un devoir de Machine Learning / Deep Learning.
