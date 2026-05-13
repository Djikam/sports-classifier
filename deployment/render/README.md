# Déploiement Render

## Configuration

1. Créer un compte sur [Render.com](https://render.com)
2. New Web Service → Connecter le repo GitHub
3. Runtime : **Docker** (détection auto du Dockerfile)
4. Plan : **Free**
5. Déployer

## Variables d'environnement (optionnel)

Dans le dashboard Render → Environment :
- `MODEL_DRIVE_URL` : lien Google Drive du modèle (format `https://drive.google.com/uc?id=FILE_ID`)

## Notes
- Premier build : ~10-15 min (installation de TensorFlow)
- Cold start : ~30s (plan gratuit)
- Limite mémoire : 512MB (suffisant pour MobileNetV2)
