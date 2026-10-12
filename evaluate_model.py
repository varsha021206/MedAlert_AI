import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("sales_history.csv")
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")

data["day"] = data["date"].dt.day
data["weekday"] = data["date"].dt.weekday
data["month"] = data["date"].dt.month

X = data[["day", "weekday", "month", "stock_available"]]
y = data["units_sold"]

with open("models/demand_model.pkl", "rb") as file:
    model = pickle.load(file)

predictions = model.predict(X)
mae = mean_absolute_error(y, predictions)

print(f"Mean Absolute Error: {mae:.2f} units per day")
print("Evaluation completed.")