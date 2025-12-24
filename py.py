import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Normal transactions
normal_data = pd.DataFrame({
    "amount": np.random.normal(2000, 800, 950),
    "hour": np.random.randint(0, 24, 950),
    "transactions_per_day": np.random.normal(3, 1, 950),
    "location_change": np.random.choice([0, 1], size=950, p=[0.9, 0.1])
})

# Fraudulent transactions (anomalies)
fraud_data = pd.DataFrame({
    "amount": np.random.normal(12000, 3000, 50),
    "hour": np.random.randint(0, 24, 50),
    "transactions_per_day": np.random.normal(15, 5, 50),
    "location_change": np.random.choice([0, 1], size=50, p=[0.2, 0.8])
})

# Combine dataset
data = pd.concat([normal_data, fraud_data], ignore_index=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
model = IsolationForest(
    n_estimators=150,
    contamination=0.05,   # Expected fraud percentage
    random_state=42
)

model.fit(scaled_data)
data["fraud_prediction"] = model.predict(scaled_data)

# Convert output
# -1 = Fraud, 1 = Normal
data["fraud_prediction"] = data["fraud_prediction"].map({1: 0, -1: 1})
fraud_count = data["fraud_prediction"].value_counts()
print(fraud_count)
plt.figure(figsize=(8,5))
plt.scatter(
    data["amount"],
    data["transactions_per_day"],
    c=data["fraud_prediction"],
    cmap="coolwarm",
    alpha=0.7
)
plt.xlabel("Transaction Amount")
plt.ylabel("Transactions Per Day")
plt.title("Credit Card Fraud Detection (Red = Fraud)")
plt.show()
new_transaction = pd.DataFrame({
    "amount": [15000],
    "hour": [3],
    "transactions_per_day": [18],
    "location_change": [1]
})

new_scaled = scaler.transform(new_transaction)
prediction = model.predict(new_scaled)

if prediction[0] == -1:
    print("🚨 Fraudulent Transaction Detected!")
else:
    print("✅ Transaction is Normal")
