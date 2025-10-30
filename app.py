# app.py
from fastapi import FastAPI
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Gradient Boosting Expense Prediction API")

# Load trained model
model = joblib.load("gb_model.pkl")

# Define start date (used for converting dates into day index)
TRAIN_START_DATE = datetime(2024, 1, 1)

@app.get("/")
def root():
    return {"message": "Gradient Boosting Expense Prediction API is running!"}

@app.get("/predict_month")
def predict_month(year: int, month: int):
    """
    Example: /predict_month?year=2025&month=11
    Returns daily and total predicted expenses for that month.
    """
    try:
        # 1️⃣ Generate all days for the given month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)

        date_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D')

        # 2️⃣ Convert each date into numeric day index
        day_indices = [(date - TRAIN_START_DATE).days + 1 for date in date_range]

        # 3️⃣ Predict using the trained model
        predictions = model.predict(pd.DataFrame({"day": day_indices}))

        # 4️⃣ Combine into result DataFrame
        result_df = pd.DataFrame({
            "date": date_range.strftime("%Y-%m-%d"),
            "predicted_expense": predictions
        })

        # 5️⃣ Calculate total monthly expense
        total_monthly_expense = float(result_df["predicted_expense"].sum())

        # 6️⃣ Return result as JSON
        return {
            "year": year,
            "month": month,
            "days_count": len(result_df),
            "total_predicted_expense": round(total_monthly_expense, 2),
            "daily_predictions": result_df.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}
