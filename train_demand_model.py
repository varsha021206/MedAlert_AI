import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load historical sales data
data = pd.read_csv("sales_history.csv")
data["date"] = pd.to_datetime(data["date"])

# Create features for the model
data["day"] = data["date"].dt.day
data["weekday"] = data["date"].dt.weekday
data["month"] = data["date"].dt.month

features = ["day", "weekday", "month", "stock_available"]
X = data[features]
y = data["units_sold"]

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train the ML model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate predictions
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# Save the model
os.makedirs("models", exist_ok=True)

with open("models/demand_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Demand model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f} units")
print("Model saved to models/demand_model.pkl")