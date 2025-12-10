import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ============================
# 1️⃣ Load dataset
# ============================
df = pd.read_csv("../Gambia_data/gambia_antibiogram_all_samples.csv", sep=';')
df.columns = df.columns.str.strip()
df = df.replace(r'^\s*$', 'NA', regex=True)

# Antibiotics to include
antibiotics = ['PEN','AMP','SXT','CN','CHL','TET','CIP','CXM',
               'ERY','PB','OX','FOX','CTX','CAZ','VA','AMC','F300']

# Mapping for binary classification
binary_map = {'S':0, 'I':0, 'R':1, 'NA':0}

# ============================
# 2️⃣ Prepare results storage
# ============================
performance_results = {}
skipped_antibiotics = []

# ============================
# 3️⃣ Train one model per antibiotic
# ============================
for ab in antibiotics:
    if ab not in df.columns:
        print(f"[] Colonne manquante : {ab} → skip.")
        skipped_antibiotics.append(ab)
        continue

    # Features = all antibiotics except target
    features = [a for a in antibiotics if a != ab]

    # Map to binary
    X = df[features].apply(lambda col: col.map(lambda x: binary_map.get(x, 0)))
    y = df[ab].map(lambda x: binary_map.get(x, 0))

    # Vérification distribution des classes
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        print(f"[] Pas assez de classes pour {ab} → skip.")
        skipped_antibiotics.append(ab)
        continue
    if min(class_counts) < 5:
        print(f"[] Classes trop déséquilibrées pour {ab} → skip.")
        skipped_antibiotics.append(ab)
        continue

    print(f"\n========================")
    print(f" Training model for: {ab}")
    print(f"Classes distribution:\n{class_counts}")
    print(f"========================\n")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"ROC-AUC: {roc:.3f}")

    # Save metrics
    performance_results[ab] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc,
        "class_distribution": class_counts.to_dict()
    }

    # Save model
    joblib.dump(model, f"../Gambia_data/model_binary_{ab}.pkl")

    # Confusion matrix (graph)
    plt.figure(figsize=(5,4))
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Non résistant", "Résistant"],
                yticklabels=["Non résistant", "Résistant"])
    plt.title(f"Matrice de Confusion - {ab}")
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.savefig(f"../Gambia_data/confusion_matrix_{ab}.png")
    plt.close()

# ============================
# 4️⃣ Save summary as JSON
# ============================
summary = {
    "performance_results": performance_results,
    "skipped_antibiotics": skipped_antibiotics
}

with open("../Gambia_data/model_performance_binary_pro.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n Tous les modèles valides ont été entraînés avec succès !")
print(f" Modèles enregistrés : model_binary_[AB].pkl")
print(f" Matrices de confusion sauvegardées.")
print(f" Résumé JSON : model_performance_binary_pro.json")
if skipped_antibiotics:
    print(f" Antibiotiques ignorés : {skipped_antibiotics}")
 