FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crée le dossier models (VIDE au départ)
RUN mkdir -p models

# Copie du code source
COPY . .

# Expose le port (Render utilise $PORT)
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Commande de démarrage
CMD streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0