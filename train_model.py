# train_model.py
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Generate sample data for training
data = pd.DataFrame({
    "date": pd.date_range(start="2024-01-01", periods=365, freq='D'),
    "total_expense": 2000 + 10 * pd.Series(range(365)) + pd.Series(range(365)).apply(lambda x: x % 7 * 5)
})

# Create numeric feature
data["day"] = range(1, len(data) + 1)

# Split features and target
X = data[["day"]]
y = data["total_expense"]

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Save trained model
joblib.dump(model, "gb_model.pkl")
print("Gradient Boosting model trained and saved as gb_model.pkl")
