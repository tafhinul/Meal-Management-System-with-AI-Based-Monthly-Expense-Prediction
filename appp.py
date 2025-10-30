# app.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

app = Flask(__name__)

# ========== 1️⃣ TRAINING FUNCTION ==========
def train_model():
    file_path = "expenses.csv"  # Your dataset file

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")

    df = pd.read_csv(file_path)

    # Inspect columns
    print("Columns in dataset:", df.columns.tolist())

    # Drop NaN
    df = df.dropna()

    # Change this according to your dataset
    # Example: predict next month’s total expense based on monthly data
    X = df.drop(columns=['TotalExpense'], errors='ignore')
    y = df['TotalExpense']

    # If non-numeric columns exist, convert them
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✅ Model trained successfully — MAE: {mae:.2f}")

    # Save model and columns
    joblib.dump(model, "model.pkl")
    joblib.dump(X.columns.tolist(), "model_columns.pkl")

    return model


# ========== 2️⃣ LOAD MODEL ==========
def load_model():
    if os.path.exists("model.pkl") and os.path.exists("model_columns.pkl"):
        model = joblib.load("model.pkl")
        columns = joblib.load("model_columns.pkl")
        print("✅ Model loaded successfully.")
        return model, columns
    else:
        print("Warning: No trained model found. Training now...")
        return train_model(), None


model, model_columns = load_model()


# ========== 3️⃣ API ROUTES ==========

@app.route('/')
def home():
    return jsonify({
        "message": "Meal Management Prediction API is running ✅",
        "routes": ["/train", "/predict"]
    })


# --- Route 1: Train model again ---
@app.route('/train', methods=['POST'])
def train():
    global model, model_columns
    model = train_model()
    model_columns = joblib.load("model_columns.pkl")
    return jsonify({"status": "Model retrained successfully"})


# --- Route 2: Predict ---
@app.route('/predict', methods=['POST'])
def predict():
    global model, model_columns
    data = request.get_json()

    if not model:
        return jsonify({"error": "Model not loaded"}), 400

    if not model_columns:
        model_columns = joblib.load("model_columns.pkl")

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return jsonify({"predicted_expense": round(float(prediction), 2)})


# ========== 4️⃣ RUN SERVER ==========
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
