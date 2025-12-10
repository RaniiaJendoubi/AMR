import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Prédiction Résistance Antimicrobienne",
    layout="wide"
)
st.title("🧬 Prédiction de la résistance bactérienne")

# =========================
# Dossiers et modèles
# =========================
MODELS_DIR = "../Gambia_data/"
antibiotics = ['PEN','AMP','SXT','CN','CHL','TET','CIP','CXM',
               'ERY','PB','OX','FOX','CTX','CAZ','VA','AMC','F300']

# Charger les modèles
models = {}
for ab in antibiotics:
    model_path = os.path.join(MODELS_DIR, f"model_binary_{ab}.pkl")
    if os.path.exists(model_path):
        models[ab] = joblib.load(model_path)

# Mapping binaire
binary_map = {'S':0, 'I':0, 'R':1, 'NA':0}

# =========================
# Upload CSV utilisateur
# =========================
st.sidebar.header("Importer un fichier CSV")
uploaded_file = st.sidebar.file_uploader(
    "Choisissez un fichier CSV contenant les antibiogrammes",
    type=["csv"]
)

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données importées")
    st.dataframe(df_input.head())

    # =========================
    # Prétraitement automatique des colonnes
    # =========================
    rename_dict = {
        'CHI': 'CHL',
        'FRY': 'ERY',
        'PR': 'PB',
        # ajouter d'autres si nécessaire
    }
    df_input.rename(columns=rename_dict, inplace=True)

    # =========================
    # Prédictions
    # =========================
    predictions = {}
    probabilities = {}

    for ab, model in models.items():
        features = [a for a in antibiotics if a != ab]

        # Vérifier colonnes manquantes
        missing_cols = [c for c in features if c not in df_input.columns]
        if missing_cols:
            st.warning(f"{ab} → Colonnes manquantes : {missing_cols}, skip")
            continue

        X = df_input[features].applymap(lambda x: binary_map.get(str(x), 0))
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:,1]

        predictions[ab] = y_pred
        probabilities[ab] = y_prob

    # =========================
    # Affichage des résultats
    # =========================
    st.subheader("📊 Résultats des prédictions")
    results_df = df_input.copy()
    for ab in predictions:
        results_df[f"{ab}_Pred"] = predictions[ab]
        results_df[f"{ab}_Prob"] = probabilities[ab]

    st.dataframe(results_df.head())

    # =========================
    # Télécharger les résultats
    # =========================
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les résultats CSV",
        data=csv,
        file_name='predictions_antibiogrammes.csv',
        mime='text/csv'
    )

    # =========================
    # Affichage matrice de confusion
    # =========================
    if predictions:
        st.subheader("🔹 Matrices de confusion (exemple)")
        selected_ab = st.selectbox("Choisir un antibiotique pour la matrice de confusion", list(predictions.keys()))
        if selected_ab:
            y_true = df_input[selected_ab].map(binary_map)
            y_pred = predictions[selected_ab]
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non résistant", "Résistant"],
                        yticklabels=["Non résistant", "Résistant"], ax=ax)
            ax.set_xlabel("Prédiction")
            ax.set_ylabel("Réel")
            ax.set_title(f"Matrice de Confusion - {selected_ab}")
            st.pyplot(fig)

 