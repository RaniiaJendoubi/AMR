import pandas as pd

df = pd.read_csv("../Gambia_data/gambia_antibiogram_sheet1.csv")

print("Aperçu des données :")
print(df.head())

df = df.dropna()



df.to_csv("../Gambia_data/gambia_preprocessed_for_model.csv", index=False)
print("\nDataset prétraité sauvegardé sous : gambia_preprocessed_for_model.csv")
