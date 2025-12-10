import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

df_all = pd.read_csv("../Gambia_data/gambia_antibiogram_all_samples.csv", sep=';')
df_all.columns = df_all.columns.str.strip()

antibiotics = ['PEN','AMP','SXT','CN','CHL','TET','CIP','CXM',
               'ERY','PB','OX','FOX','CTX','CAZ','VA','AMC','F300']
df_all = df_all[antibiotics]

mapping = {'S':0, 'I':1, 'R':2, 'NA':0}

for ab in antibiotics:
    print(f"\n=== Validation du modèle pour {ab} ===")

    try:
        model_path = f"../Gambia_data/random_forest_model_{ab}.pkl"
        model = joblib.load(model_path)
        print(f"Modèle pour {ab} chargé avec succès !")

        features = [a for a in antibiotics if a != ab]
        
        df_all = df_all[antibiotics]
        X_all = df_all[features].apply(lambda col: col.map(lambda x: mapping[x] if x in mapping else 0))
        y_all = df_all[ab].map(lambda x: mapping[x] if x in mapping else 0)

        if y_all.nunique() < 2:
            print(f"Pas assez de classes pour {ab}, skip.")
            continue

        y_pred = model.predict(X_all)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_all, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_all, y_pred))

    except FileNotFoundError:
        print(f"Modèle pour {ab} non trouvé — passe au suivant.")
    except Exception as e:
        print(f" Erreur avec {ab} : {e}")
