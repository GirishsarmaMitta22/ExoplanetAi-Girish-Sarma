import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 

# --- 1. Data Loading (Module 1: Complete) ---

def load_existing_data(file_name='PS_2025.12.04_05.00.24.csv'):
    """
    Loads the existing local CSV file, skipping 96 rows of NASA metadata.
    """
    HEADER_ROW = 96
    try:
        df = pd.read_csv(file_name, header=HEADER_ROW)
        return df
    except Exception as e:
        print(f"Module 1 CRITICAL ERROR: Could not load the local file. Check file name and location. Error: {e}")
        return None

# --- 2. Custom Feature Transformers (Module 2: Feature Engineering) ---

class HabitabilityIndexTransformer(BaseEstimator, TransformerMixin):
    """Calculates the composite Habitability Score Index (Proxy)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        mass = X_copy['pl_bmasse'].fillna(X_copy['pl_bmasse'].median())
        radius = X_copy['pl_rade'].fillna(X_copy['pl_rade'].median())
        eqt = X_copy['pl_eqt'].fillna(X_copy['pl_eqt'].median())
        
        X_copy['Habitability_Index'] = (eqt / 255.0) * (radius / 1.5)
        
        return X_copy[['Habitability_Index']]

class StellarCompatibilityIndexTransformer(BaseEstimator, TransformerMixin):
    """Calculates the composite Stellar Compatibility Index (Proxy)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        teff = X_copy['st_teff'].fillna(X_copy['st_teff'].median())
        X_copy['Stellar_Compatibility_Index'] = teff / 5778.0
        
        return X_copy[['Stellar_Compatibility_Index']]

# --- 3. Define Preprocessing Pipeline (Module 2: Cleaning) ---

def get_feature_names(preprocessor, numerical_cols, cat_cols, index_names):
    """Retrieves the full list of column names after all transformations."""
    
    feature_names = list(numerical_cols)
    
    ohe_transformer = preprocessor.named_transformers_['cat'].named_steps['onehot']
    
    cat_feature_names = list(ohe_transformer.get_feature_names_out(cat_cols))
    feature_names.extend(cat_feature_names)
    
    feature_names.extend(index_names) 
    
    return feature_names


def create_preprocessing_pipeline():
    """Defines the ColumnTransformer for cleaning, scaling, and engineering."""
    
    NUMERICAL_FEATURES = ['pl_bmasse', 'pl_rade', 'pl_eqt', 'pl_orbper', 'st_teff', 'st_mass', 'st_met']
    CATEGORICAL_FEATURES = ['st_spectype']
    INDEX_FEATURE_NAMES = ['Habitability_Index', 'Stellar_Compatibility_Index']

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()) 
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    full_pipeline = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
            ('hab_idx', HabitabilityIndexTransformer(), ['pl_bmasse', 'pl_rade', 'pl_eqt']),
            ('stel_idx', StellarCompatibilityIndexTransformer(), ['st_teff']) 
        ],
        remainder='drop'
    )
    
    return full_pipeline, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, INDEX_FEATURE_NAMES

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    # --- TRACE 1: MODIFIED OUTPUT ---
    print("--- TRACE 1: Starting Data Load ---") 
    
    df = load_existing_data()
    
    if df is not None:
        
        preprocessor, NUM_F, CAT_F, INDEX_F = create_preprocessing_pipeline()
        
        print("--- TRACE 2: Pipeline Definition Success. Starting Execution ---")

        # Prepare Target Variable
        df['Habitability_Class'] = np.random.randint(0, 2, size=len(df))
        X = df.drop('Habitability_Class', axis=1, errors='ignore')

        try:
            # Apply the entire cleaning and engineering process (Module 2)
            X_processed_array = preprocessor.fit_transform(X) 
            
            # Get the correct column names 
            final_columns = get_feature_names(preprocessor, NUM_F, CAT_F, INDEX_F)
            
            # Convert the processed NumPy array back into a DataFrame WITH HEADERS
            X_processed_df = pd.DataFrame(X_processed_array, columns=final_columns)
            
            OUTPUT_FILE = 'exoplanet_data_processed.csv'
            X_processed_df.to_csv(OUTPUT_FILE, index=False)
            
            # --- FINAL SUCCESS MESSAGE (MODIFIED) ---
            print("\n--- Project Foundation Completed ---")
            print("Execution complete.") 
            print(f"Output File: {OUTPUT_FILE} created with {X_processed_df.shape[1]} named columns.")

        except KeyError as e:
            print("\n--- CRITICAL ERROR: PIPELINE FAILED (COLUMN MISSING) ---")
            print(f"The program crashed because the column {e} is not in the CSV file.")
            print("ACTION REQUIRED: Check the spelling of this column in the 'create_preprocessing_pipeline' function.")
        except Exception as e:
            print(f"\n--- CRITICAL ERROR: UNKNOWN FAILURE ---")
            print(f"The script crashed during pipeline execution. **Error:** {e}")
