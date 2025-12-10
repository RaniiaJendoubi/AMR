import joblib
import pandas as pd

model = joblib.load("../Gambia_data/model_binary_PEN.pkl")

row = pd.DataFrame([{
    "AMP":0, "SXT":1, "CN":0, "CHL":1, "TET":0,
    "CIP":1, "CXM":0, "ERY":1, "PB":0, "OX":1,
    "FOX":0, "CTX":1, "CAZ":0, "VA":1, "AMC":0, "F300":1
}])

print("Prediction:", model.predict(row)[0])
print("Probabilité résistance :", model.predict_proba(row)[0][1])
