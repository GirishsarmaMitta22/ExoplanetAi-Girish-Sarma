import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
print("--- Step 1: Loading Data ---")
# header=96 is used to skip the NASA introduction text
df = pd.read_csv('PS_2025.12.04_05.00.24.csv', header=96)

# Selecting simple columns for our study
cols_to_use = ['pl_bmasse', 'pl_rade', 'pl_eqt', 'pl_orbper', 'st_teff', 'st_mass', 'st_met', 'st_spectype']
df = df[cols_to_use]

# --- 2. HANDLE MISSING VALUES ---
print("--- Step 2: Filling Empty Cells ---")
# For numbers, we find the middle value (median) and fill the gaps
df['pl_bmasse'] = df['pl_bmasse'].fillna(df['pl_bmasse'].median())
df['pl_rade'] = df['pl_rade'].fillna(df['pl_rade'].median())
df['pl_eqt'] = df['pl_eqt'].fillna(df['pl_eqt'].median())
df['pl_orbper'] = df['pl_orbper'].fillna(df['pl_orbper'].median())
df['st_teff'] = df['st_teff'].fillna(df['st_teff'].median())
df['st_mass'] = df['st_mass'].fillna(df['st_mass'].median())
df['st_met'] = df['st_met'].fillna(df['st_met'].median())

# For the star type (text), we find the most frequent one
most_common_star = df['st_spectype'].mode()[0]
df['st_spectype'] = df['st_spectype'].fillna(most_common_star)

# --- 3. FEATURE ENGINEERING ---
print("--- Step 3: Creating New Metrics ---")
# Simple math to create our indices
df['Habitability_Index'] = (df['pl_eqt'] / 255.0) * (df['pl_rade'] / 1.5)
df['Stellar_Compatibility_Index'] = df['st_teff'] / 5778.0

# --- 4. ONE-HOT ENCODING ---
print("--- Step 4: Changing Text to Numbers ---")
df_encoded = pd.get_dummies(df, columns=['st_spectype'])                                           # Convert 'st_spectype' into 0s and 1s so the AI can read it

# --- 5. NORMALIZATION (Scaling) ---
print("--- Step 5: Scaling Numbers ---")
# We want all numbers to be in a similar small range
numeric_columns = df_encoded.select_dtypes(include=[np.number]).columns                            # Formula: (Value - Mean) / Standard Deviation

for col in numeric_columns:
    col_mean = df_encoded[col].mean()
    col_std = df_encoded[col].std()
    if col_std != 0:
        df_encoded[col] = (df_encoded[col] - col_mean) / col_std

# --- 6. VALIDATION & EXPORT ---
print("--- Step 6: Final Check & Saving ---")
print(df_encoded[['Habitability_Index', 'Stellar_Compatibility_Index']].describe())           # Print descriptive statistics (Validation)

df_encoded['Habitability_Index'].hist(bins=20)
plt.title('Habitability Index Check')                                                           # Save a histogram (Visualization)
plt.savefig('validation_chart.png')

df_encoded.to_csv('exoplanet_data_processed.csv', index=False)

