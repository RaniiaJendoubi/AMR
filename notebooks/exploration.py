import pandas as pd

df = pd.read_csv(r"C:\Users\ranii\antimicrobial_resistance_project\Gambia_data\gambia_antibiogram_sheet1.csv")

print("\nAperçu des données :")
print(df.head())

print("\nInfos générales :")
print(df.info())

print("\nStatistiques descriptives :")
print(df.describe())

print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

df.to_csv("C:/Users/ranii/antimicrobial_resistance_project/Gambia_data/gambia_sheet1_preprocessed_copy.csv", index=False)
print("\nDataset prétraité sauvegardé sous : gambia_sheet1_preprocessed_copy.csv")
