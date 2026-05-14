FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crée le dossier models
RUN mkdir -p models

# Copie du code source
COPY . .

# Télécharge le modèle AU BUILD (pas au runtime)
# Remplace par ton URL GitHub Releases quand tu l'auras
ARG MODEL_URL=https://drive.google.com/uc?id=1S46qADbZHvMNOf13_1-WLvhLGnY4RqN-&export=download&confirm=t
RUN curl -L -o models/best_model.h5 \
    -H "User-Agent: Mozilla/5.0" \
    "$MODEL_URL" || echo "⚠️ Téléchargement échoué, sera retenté au runtime"

# Vérifie le fichier téléchargé
RUN python -c "
import os
if os.path.exists('models/best_model.h5'):
    with open('models/best_model.h5', 'rb') as f:
        header = f.read(8)
    if header[:8] == b'\x89HDF\r\n\x1a\n':
        print('✅ Modèle HDF5 valide au build')
    else:
        print('⚠️ Format invalide au build, suppression...')
        os.remove('models/best_model.h5')
"

# Expose le port (Render remplace par $PORT au runtime)
EXPOSE 8501

# Healthcheck sur le bon port
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Commande de démarrage avec $PORT
CMD streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0