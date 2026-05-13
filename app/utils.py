"""
Utilitaires pour l'application Streamlit.
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_probs(results):
    """
    Affiche un graphique à barres des probabilités top-k.

    Args:
        results: liste de tuples (nom_classe, probabilité)
    """
    classes = [r[0] for r in results]
    probs = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
    bars = ax.barh(classes[::-1], probs[::-1], color=colors[::-1])

    ax.set_xlabel('Probabilité')
    ax.set_title('Top Prédictions')
    ax.set_xlim(0, 1)

    # Ajouter les valeurs sur les barres
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.2%}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def display_metrics():
    """Affiche des métriques factices/plateholders dans la sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Métriques du Modèle")
    st.sidebar.metric("Accuracy (valid)", "93.2%", "+2.1%")
    st.sidebar.metric("Accuracy (test)", "91.5%", "+1.8%")
    st.sidebar.metric("Classes", "100", None)
    st.sidebar.metric("Modèle", "MobileNetV2", None)
