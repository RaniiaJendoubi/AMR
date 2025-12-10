import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

df_all = pd.read_csv("../Gambia_data/gambia_antibiogram_all_samples.csv", sep=';')
print("Colonnes du dataset :")
print(df_all.columns.tolist())

df_all.columns = df_all.columns.str.strip()  
df_all = df_all.replace(r'^\s*$', 'NA', regex=True)  

antibiotics = ['PEN','AMP','SXT','CN','CHL','TET','CIP','CXM',
               'ERY','PB','OX','FOX','CTX','CAZ','VA','AMC','F300']

mapping = {'S':0, 'I':1, 'R':2, 'NA':0}

results_df = pd.DataFrame(columns=['Antibiotic', 'Accuracy'])

for ab in antibiotics:
    if ab not in df_all.columns:
        print(f"Colonne manquante : {ab}, skip.")
        continue
    
    features = [a for a in antibiotics if a != ab]
    X = df_all[features].apply(lambda col: col.map(lambda x: mapping[x] if x in mapping else 0))
    y = df_all[ab].map(lambda x: mapping[x] if x in mapping else 0)
    
    if y.nunique() < 2:
        print(f"Pas assez de classes pour {ab}, skip.")
        continue
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
   
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"=== {ab} : Accuracy = {acc:.2f}")
    
    joblib.dump(model, f"../Gambia_data/random_forest_model_{ab}.pkl")
    
    results_df = pd.concat([results_df, pd.DataFrame({'Antibiotic':[ab], 'Accuracy':[acc]})], ignore_index=True)

print("\n=== Résumé des modèles ===")
print(results_df)

plt.figure(figsize=(12,6))
plt.bar(results_df['Antibiotic'], results_df['Accuracy'], color='skyblue')
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.title('Précision des modèles RandomForest par antibiotique')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
