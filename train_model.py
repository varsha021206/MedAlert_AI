import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

os.makedirs("models", exist_ok=True)

data = pd.DataFrame({
    "day": list(range(1, 31)) * 3,
    "month": [1] * 90,
    "stock": [100, 95, 92, 88, 85, 80, 78, 75, 70, 68] * 9,
    "demand": [
        8, 10, 9, 12, 11, 14, 13, 15, 12, 16,
        14, 17, 15, 18, 16, 19, 17, 20, 18, 21,
        19, 22, 20, 23, 21, 24, 22, 25, 23, 26
    ] * 3
})

X = data[["day", "month", "stock"]]
y = data["demand"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

with open("models/demand_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Demand forecasting model trained and saved!")