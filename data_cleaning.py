//This code defines the structure necessary to clean, scale, encode, & create our custom indices within an efficient Scikit-learn workflow.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class HabitabilityIndexTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, mass_col='pl_masse', radius_col='pl_rade', temp_col='pl_eqt'):
        self.mass_col = mass_col
        self.radius_col = radius_col
        self.temp_col = temp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        density = X_copy[self.mass_col] / (4/3 * np.pi * (X_copy[self.radius_col]**3))
        
        X_copy['Habitability_Index'] = (X_copy[self.temp_col] * np.log1p(X_copy[self.mass_col])) / (density + 1)
        
        return X_copy[['Habitability_Index']]

class StellarCompatibilityIndexTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, teff_col='st_teff', lum_col='st_lum'):
        self.teff_col = teff_col
        self.lum_col = lum_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        X_copy['Stellar_Compatibility_Index'] = X_copy[self.lum_col] / (X_copy[self.teff_col]**2)
        
        return X_copy[['Stellar_Compatibility_Index']]


NUMERICAL_FEATURES = [
    'pl_rade', 'pl_masse', 'pl_orbper', 'pl_eqt', 'st_teff', 'st_lum', 'st_met', 'st_mass'
]
CATEGORICAL_FEATURES = [
    'st_spectype'
]
HABITABILITY_INDEX_INPUTS = ['pl_masse', 'pl_rade', 'pl_eqt']
STELLAR_INDEX_INPUTS = ['st_teff', 'st_lum']


numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) 
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])


feature_engineering_pipeline = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, NUMERICAL_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ('hab_idx', HabitabilityIndexTransformer(), HABITABILITY_INDEX_INPUTS),
        ('stel_idx', StellarCompatibilityIndexTransformer(), STELLAR_INDEX_INPUTS)
    ],
    remainder='drop'
)

if __name__ == '__main__':
    
    data = pd.DataFrame({
        'pl_rade': [1.5, 2.0, np.nan, 1.1],
        'pl_masse': [5.0, np.nan, 6.0, 4.5],
        'pl_eqt': [250, 300, 290, np.nan],
        'st_teff': [5000, 4500, np.nan, 5200],
        'st_lum': [0.5, 0.2, 0.4, np.nan],
        'pl_orbper': [100, 50, 200, 150],
        'st_met': [0.1, 0.0, 0.2, 0.15],
        'st_mass': [0.8, 0.7, 0.9, 0.75],
        'st_spectype': ['G', 'M', 'G', 'K'],
        'Habitability_Class': [1, 0, 1, 0]
    })
    
    X = data.drop('Habitability_Class', axis=1)
    
    X_processed = feature_engineering_pipeline.fit_transform(X)
    
    print(X_processed[:5])
