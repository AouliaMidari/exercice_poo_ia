import streamlit as st
import sys
import os

# Permet d'importer les dossiers parents (core, app, etc.)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.logistic_regression import LogisticRegressionModel
from core.neural_network import NeuralNetworkModel
from app.interface_clinique import ClinicalPredictor

# --- 1. CONFIGURATION DE LA PAGE & CSS ---
st.set_page_config(page_title="Diagnostic IA", page_icon="🏥", layout="centered")

# Injection de CSS personnalisé pour faire "pro"
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        width: 100%;
        height: 50px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .infected { background-color: #ffcccc; color: #cc0000; border: 2px solid #cc0000; }
    .healthy { background-color: #ccffcc; color: #006600; border: 2px solid #006600; }
    </style>
""", unsafe_allow_html=True)

# --- 2. INTERFACE GRAPHIQUE ---
st.title("🏥 Système de Diagnostic Clinique IA")
st.write("Remplissez les informations du patient pour obtenir une prédiction en temps réel.")

# Choix du modèle (pour prouver le polymorphisme à l'oral !)
st.sidebar.header("⚙️ Paramètres du Modèle")
choix_modele = st.sidebar.selectbox(
    "Choisissez le modèle d'IA :",
    ("Régression Logistique", "Réseau de Neurones")
)

# Formulaire de saisie des données
with st.form("patient_form"):
    st.subheader("📋 Données du Patient")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input("Température (°C)", min_value=34.0, max_value=42.0, value=37.0, step=0.1)
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)
        
    with col2:
        toux = st.selectbox("Toux ?", ("Non", "Oui"))
        fatigue = st.selectbox("Fatigue ?", ("Non", "Oui"))
    
    # Bouton de soumission
    submit_button = st.form_submit_button("Lancer le diagnostic")

# --- 3. LOGIQUE MÉTIER (Lien avec votre POO) ---
if submit_button:
    # 1. Préparation des données
    patient_data = [
        temperature,
        1 if toux == "Oui" else 0,
        1 if fatigue == "Oui" else 0,
        age
    ]
    
    # 2. Sélection du modèle (Polymorphisme)
    if choix_modele == "Régression Logistique":
        modele = LogisticRegressionModel()
    else:
        modele = NeuralNetworkModel()
        
    # 3. Utilisation de votre classe ClinicalPredictor !
    predictor = ClinicalPredictor(trained_model=modele)
    resultat = predictor.diagnose(patient_data)
    
    # 4. Affichage du résultat avec style
    st.markdown("---")
    st.subheader("Résultat du Diagnostic :")
    
    if resultat == "Infecté":
        st.markdown(f'<div class="result-box infected">⚠️ {resultat}</div>', unsafe_allow_html=True)
        st.error("Veuillez consulter un médecin immédiatement.")
    else:
        st.markdown(f'<div class="result-box healthy">✅ {resultat}</div>', unsafe_allow_html=True)
        st.success("Le patient semble en bonne santé.")