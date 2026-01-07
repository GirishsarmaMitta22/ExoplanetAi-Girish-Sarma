from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import sqlite3
import os

app = Flask(__name__)

# --- CONFIGURATION ---
DB_NAME = 'exoplanets.db'
API_KEY = "secret_api_key_123"  # Simple security for demo

# --- LOAD MODELS ---
try:
    rf_model = joblib.load('rf_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    rf_model = None
    xgb_model = None

# --- DATABASE HELPER ---
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# --- SECURITY ---
def check_auth(request):
    key = request.headers.get('x-api-key')
    return key == API_KEY

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Exoplanet Habitability Prediction API", "status": "running"})

@app.route('/predict', methods=['POST'])
def predict():
    if not check_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    if not rf_model or not xgb_model:
        return jsonify({"error": "Models not loaded"}), 500

    try:
        data = request.get_json()
        
        # Expecting a dictionary of features matching the training data
        # For simplicity, we assume the input JSON keys match the model's feature names
        # In a real app, you'd likely map generic inputs to specific model features
        
        # Create DataFrame from input (single row)
        input_df = pd.DataFrame([data])
        
        # Ensure columns match training data (fill missing with 0 or handle appropriately)
        # This is a simplified approach. Ideally, you align with training columns.
        model_cols = xgb_model.get_booster().feature_names
        
        # Add missing columns with 0
        for col in model_cols:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Reorder to match model's expected input
        input_df = input_df[model_cols]
        
        # Predictions
        rf_pred = int(rf_model.predict(input_df)[0])
        xgb_pred = int(xgb_model.predict(input_df)[0])
        
        # Probabilities for ranking
        rf_proba = rf_model.predict_proba(input_df)[0][1]
        xgb_proba = xgb_model.predict_proba(input_df)[0][1]
        
        # Averaged Habitability Score (0 to 1)
        habitability_score = (rf_proba + xgb_proba) / 2
        
        return jsonify({
            "prediction": {
                "random_forest": rf_pred,
                "xgboost": xgb_pred,
                "consensus": 1 if habitability_score > 0.5 else 0
            },
            "habitability_score": round(habitability_score, 4),
            "ranking": "High" if habitability_score > 0.8 else "Moderate" if habitability_score > 0.5 else "Low"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/exoplanets', methods=['GET'])
def get_exoplanets():
    limit = request.args.get('limit', 10)
    offset = request.args.get('offset', 0)
    
    try:
        conn = get_db_connection()
        query = f"SELECT * FROM exoplanets LIMIT {limit} OFFSET {offset}"
        data = conn.execute(query).fetchall()
        conn.close()
        
        # Convert rows to dicts
        results = [dict(row) for row in data]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
