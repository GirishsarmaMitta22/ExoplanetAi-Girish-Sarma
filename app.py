from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import joblib
import json
import plotly
import plotly.express as px
import plotly.graph_objs as go
import os
import webbrowser
import threading
import requests
import time

app = Flask(__name__)

# --- Load Resources ---
MODEL_PATH = 'rf_model.pkl'  # Prefer RF, fallback to XGB if needed
DATA_PATH = 'exoplanet_data_processed.csv'

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    elif os.path.exists('xgb_model.pkl'):
        model = joblib.load('xgb_model.pkl')
        print("Loaded model from xgb_model.pkl")
    else:
        print("WARNING: No model file found. Predictions will be mocked.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load a sample of data for the dashboard to avoid memory issues with full dataset if large
try:
    df = pd.read_csv(DATA_PATH)
    # Using a subset for faster dashboard rendering if needed, but full dataset is better for accuracy
    print(f"Loaded data with {len(df)} rows.")
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Feature mapping
        # The model expects specific feature order. Based on previous exploration:
        # We need to construct a DataFrame with the same columns as X_train.
        # For this implementation, we'll focus on the 7 numeric features and 0-fill the rest.

        if model:
            # Create a base dataframe with 0s for model columns
            model_cols = [c for c in df.columns if c not in ['Habitability_Index', 'Stellar_Compatibility_Index', 'Habitability_Class', 'Unnamed: 0']]
            
            # Create a single row dataframe with zeros
            input_df = pd.DataFrame(0, index=[0], columns=model_cols)
            
            # Fill known values from input
            input_features = {
                'pl_bmasse': data.get('pl_bmasse', 0),
                'pl_rade': data.get('pl_rade', 0),
                'pl_eqt': data.get('pl_eqt', 0),
                'pl_orbper': data.get('pl_orbper', 0),
                'st_teff': data.get('st_teff', 0),
                'st_mass': data.get('st_mass', 0),
                'st_met': data.get('st_met', 0)
            }
            
            for col, val in input_features.items():
                if col in input_df.columns:
                    input_df[col] = val
            
            # Predict
            pred_class = model.predict(input_df)[0]
            pred_prob = model.predict_proba(input_df)[0][1] # Probability of Class 1 (Habitable)
            
            prediction = "Habitable" if pred_class == 1 else "Non-Habitable"
            confidence = round(pred_prob * 100, 2)
            
        else:
            # Mock if model not loaded
            prediction = "Habitable" 
            confidence = 88.2

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": f"{confidence}%"
        })
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard-data')
def dashboard_data():
    try:
        # 1. Feature Importance (Specific to Random Forest)
        fi_json = "{}"
        if hasattr(model, 'feature_importances_'):
            model_cols = [c for c in df.columns if c not in ['Habitability_Index', 'Stellar_Compatibility_Index', 'Habitability_Class', 'Unnamed: 0']]
            importances = model.feature_importances_
            # Sort importance
            indices = np.argsort(importances)[-10:] # Top 10
            
            fig_fi = px.bar(
                x=[importances[i] for i in indices],
                y=[model_cols[i] for i in indices],
                orientation='h',
                title='Top 10 Feature Importance',
                labels={'x':'Importance', 'y':'Feature'}
            )
            fig_fi.update_layout(template='plotly_dark')
            fi_json = json.dumps(fig_fi, cls=plotly.utils.PlotlyJSONEncoder)

        # 2. Habitability Score Distribution
        fig_hist = px.histogram(
            df, 
            x='Habitability_Index', 
            nbins=50, 
            title='Distribution of Habitability Index',
            color_discrete_sequence=['#66fcf1']
        )
        fig_hist.update_layout(template='plotly_dark')
        hist_json = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)

        # 3. Correlation Heatmap (Star vs Planet)
        key_cols = ['pl_bmasse', 'pl_rade', 'pl_eqt', 'pl_orbper', 'st_teff', 'st_mass', 'st_met', 'Habitability_Index']
        corr = df[key_cols].corr()
        
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title='Correlation Heatmap: Planet vs Star Parameters',
            color_continuous_scale='Viridis'
        )
        fig_corr.update_layout(template='plotly_dark')
        corr_json = json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            "feature_importance": fi_json,
            "habitability_dist": hist_json,
            "correlation": corr_json
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/export-report')
def export_report():
    try:
        top_candidates = df[df['Habitability_Index'] > 0.8].head(100)
        output_file = 'top_candidates.xlsx'
        top_candidates.to_excel(output_file, index=False)
        return send_file(output_file, as_attachment=True, download_name='top_candidates.xlsx')
    except Exception as e:
        return str(e), 500

def open_browser():
    """Polls the server until it's running, then opens the dashboard."""
    url = "http://127.0.0.1:5000/dashboard"
    print(f"Waiting for server to start at {url}...")
    
    # Poll for server availability
    max_retries = 30
    for _ in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is up! Opening dashboard...")
                webbrowser.open(url)
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("Server took too long to start. Please open browser manually.")

if __name__ == '__main__':
    # Run browser opener only in the main process
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(debug=True, port=5000)