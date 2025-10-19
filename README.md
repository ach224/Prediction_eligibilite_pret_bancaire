# Projet 1 – Prédiction de l’éligibilité à un prêt bancaire
## 🎯 Objectif du projet
Prédire si un client est éligible à un prêt bancaire en utilisant des modèles de machine learning (ex. Gradient Boosting, Logistic Regression). L’enjeu est de reproduire le processus de scoring bancaire sur un dataset public.
## 📂 Dataset
Nom : Loan Prediction Dataset

Source : https://www.kaggle.com/datasets/ninzaami/loan-predication

## 🛠️ Étapes du projet
**1. Compréhension et préparation des données**
- Charger le dataset
- Comprendre les colonnes
- Vérifier valeurs manquantes, doublons, valeurs aberrantes
- Encoder variables catégorielles
- Normaliser / standardiser les variables numériques si nécessaire

**2. Analyse exploratoire (EDA)**
- Visualiser la distribution des variables
- Comparer revenus entre éligibles et non éligibles
- Étudier impact de Credit_History et Education
- Vérifier déséquilibre des classes dans Loan_Status

**3. Modélisation**
- Définir variable cible : Loan_Status
- Séparer train/test
- Tester plusieurs modèles (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Comparer performances (Accuracy, Precision, Recall, F1-score, ROC-AUC)

**4. Optimisation**
- Feature engineering (Income-to-Loan-Ratio)
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Gestion du déséquilibre (SMOTE, class_weight)

**5. Évaluation finale**
- Comparer résultats sur test set
- Sélectionner modèle final
- Interpréter features importantes (feature importance, SHAP values)

**6. Restitution**
- Rédiger un rapport clair avec objectif, méthodologie, résultats et recommandations

**7. (Optionnel) Application**
- Créer un dashboard avec Streamlit ou Gradio permettant de saisir les infos d’un client et prédire son éligibilité
