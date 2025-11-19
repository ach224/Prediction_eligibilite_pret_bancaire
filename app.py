import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Chargement du modèle et des données

@st.cache_resource
def load_model():
    with open("Modele_rf.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_resource
def load_data():
    with open("X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    return X_test, y_test


# Pages de l'application

def page_accueil():
    st.title("Application d'Éligibilité au Prêt")
    st.markdown("""
    Bienvenue dans votre application de prédiction d'éligibilité à un prêt !  
    Utilisez le menu à gauche pour naviguer entre :
    
    - 🔍 **Faire une prédiction**
    - 📊 **Évaluer les performances du modèle**
    - ℹ️ **À propos**
    """)


def page_prediction(model):
    st.title("🔍 Faire une prédiction")
    st.write("Veuillez entrer les informations du client :")

    # Utiliser un formulaire pour regrouper les inputs et le bouton de soumission
    with st.form(key='prediction_form'):
        # Variables numériques
        dependents = st.number_input("Nombre de personnes à charge", min_value=0)
        applicant_income = st.number_input("Revenu du demandeur (€)", min_value=0.0, step=0.01)
        coapplicant_income = st.number_input("Revenu du co-demandeur (€)", min_value=0.0, step=0.01)
        loan_amount = st.number_input("Montant du prêt demandé (€)", min_value=0.0, step=0.01)
        loan_term = st.number_input("Durée du prêt (en mois)", min_value=0)

        # Variables catégorielles binaires
        credit_history = st.selectbox("Historique de crédit", ["Bon", "Mauvais"])
        credit_history_encoded = 1 if credit_history == "Bon" else 0
        gender_male = int(st.checkbox("Genre : Homme"))
        married_yes = int(st.checkbox("Marié(e)"))
        education_not_graduate = int(st.checkbox("Non diplômé(e)"))
        self_employed_yes = int(st.checkbox("Travailleur indépendant"))

        # Zones géographiques
        property_area = st.selectbox("Zone de propriété", ["Rurale", "Semi-urbaine", "Urbaine"])
        property_area_semiurban = 1 if property_area == "Semi-urbaine" else 0
        property_area_urban = 1 if property_area == "Urbaine" else 0

        # Feature dérivée
        ratio_revenu_pret = (applicant_income + coapplicant_income) / loan_amount if loan_amount > 0 else 0

        # Création du DataFrame pour la prédiction
        colonnes = [
            "Dependents",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Gender_Male",
            "Married_Yes",
            "Education_Not Graduate",
            "Self_Employed_Yes",
            "Property_Area_Semiurban",
            "Property_Area_Urban",
            "ratio_revenu_pret"
        ]

        donnees = pd.DataFrame([[ 
            dependents,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history_encoded,
            gender_male,
            married_yes,
            education_not_graduate,
            self_employed_yes,
            property_area_semiurban,
            property_area_urban,
            ratio_revenu_pret
        ]], columns=colonnes)

        # Bouton de soumission
        submit_button = st.form_submit_button(label='Prédire')

    # Prédiction
    if submit_button:
        if loan_amount == 0 or loan_term == 0 or applicant_income == 0:
            st.warning("⚠️ Veuillez remplir toutes les informations obligatoires avant de prédire.")
        else:
            prediction = model.predict(donnees)[0]

            if prediction == 1:
                st.success("✅ Le client est **éligible** au prêt.")
            else:
                st.error("❌ Le client n'est **pas éligible** au prêt.")


def page_evaluation(model, X_test, y_test):
    st.title("📊 Évaluation du modèle")

    y_pred = model.predict(X_test)

    st.subheader("📄 Rapport de Classification")
    
    # Meilleure présentation
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(report_dict)

    st.subheader("🧩 Matrice de Confusion")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Valeurs réelles")
    st.pyplot(fig)


def page_about():
    st.title("ℹ️ À propos")
    st.markdown("""
    Cette application a été développée pour prédire l'éligibilité d'un client à un prêt 
    à partir d'un modèle de machine learning.

    **Technologies utilisées :**
    - Python
    - Streamlit
    - Scikit-learn
    - Pickle

    **Développées par Aissatou Lamarana Barry & Aicha Souaré.**

    """)


# Layout principal
def main():
    st.sidebar.title("📌 Navigation")

    menu = st.sidebar.radio(
        "Aller à :",
        ["🏠 Accueil", "🔍 Prédiction", "📊 Évaluation du modèle", "ℹ️ À propos"]
    )

    model = load_model()
    X_test, y_test = load_data()

    if menu == "🏠 Accueil":
        page_accueil()
    elif menu == "🔍 Prédiction":
        page_prediction(model)
    elif menu == "📊 Évaluation du modèle":
        page_evaluation(model, X_test, y_test)
    elif menu == "ℹ️ À propos":
        page_about()


if __name__ == "__main__":
    main()
