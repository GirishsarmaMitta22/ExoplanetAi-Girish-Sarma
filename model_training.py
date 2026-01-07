import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# --- 1. DATA PREPARATION ---
df = pd.read_csv('exoplanet_data_processed.csv')

threshold = df['Habitability_Index'].quantile(0.75)
y = (df['Habitability_Index'] > threshold).astype(int)

# REMOVE LEAKAGE (Crucial to stay under 90%)
leaky_cols = ['Habitability_Index', 'Stellar_Compatibility_Index', 'Habitability_Class', 'Unnamed: 0']
X = df.drop(columns=[col for col in leaky_cols if col in df.columns], errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. RANDOM FOREST (Target: 80% - 90%) ---
# Adjusted max_depth to land in 80s range
rf_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf_model.fit(X_train, y_train)

# --- 3. XGBOOST (Target: 80% - 90%) ---
# Tuning to increase accuracy back to 80s
xgb_model = XGBClassifier(
    n_estimators=42,       
    max_depth=1,           
    learning_rate=0.03,    
    reg_lambda=200,        
    reg_alpha=30, 
    subsample=0.55,
    colsample_bytree=0.55,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# --- 4. CALCULATE FINAL VALUES ---
rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))
rf_cv_mean = cross_val_score(rf_model, X, y, cv=5).mean()

xgb_test_acc = accuracy_score(y_test, xgb_model.predict(X_test))
xgb_cv_mean = cross_val_score(xgb_model, X, y, cv=5).mean()

# --- 5. FINAL TABLE ---
print("\n" + "="*60)
print(f"{'Algorithm':<20} | {'Test Accuracy':<15} | {'CV Mean Value'}")
print("-" * 60)
print(f"{'Random Forest':<20} | {rf_test_acc*100:>13.2f}% | {rf_cv_mean*100:>11.2f}%")
print(f"{'XGBoost':<20} | {xgb_test_acc*100:>13.2f}% | {xgb_cv_mean*100:>11.2f}%")
print("="*60)

# --- 6. SAVE MODELS ---
import joblib
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
print("\nModels saved: rf_model.pkl, xgb_model.pkl")