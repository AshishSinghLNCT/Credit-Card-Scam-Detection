import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

np.random.seed(42)

normal_data = pd.DataFrame({
    "amount": np.random.normal(2000, 800, 950),
    "hour": np.random.randint(0, 24, 950),
    "transactions_per_day": np.random.normal(3, 1, 950),
    "location_change": np.random.choice([0, 1], size=950, p=[0.9, 0.1])
})

fraud_data = pd.DataFrame({
    "amount": np.random.normal(12000, 3000, 50),
    "hour": np.random.randint(0, 24, 50),
    "transactions_per_day": np.random.normal(15, 5, 50),
    "location_change": np.random.choice([0, 1], size=50, p=[0.2, 0.8])
})

data = pd.concat([normal_data, fraud_data], ignore_index=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

model = IsolationForest(
    n_estimators=150,
    contamination=0.05,
    random_state=42
)
model.fit(scaled_data)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model & Scaler saved successfully")
