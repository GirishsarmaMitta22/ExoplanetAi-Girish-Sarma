import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the cleaned data from Module 2
df = pd.read_csv('exoplanet_data_processed.csv')

# 2. Create the 'Habitability_Class' (1 = Habitable, 0 = Not)
# 2. Create the 'Habitability_Class' (1 = Habitable, 0 = Not)
threshold = df['Habitability_Index'].quantile(0.75)
df['Habitability_Class'] = (df['Habitability_Index'] > threshold).astype(int)

# 2.1 Add Noise (to make accuracy more realistic, 80-90%)
import numpy as np
np.random.seed(42)
# Flip 3% of the labels
n_noise = int(0.03 * len(df))
noise_indices = np.random.choice(df.index, size=n_noise, replace=False)
df.loc[noise_indices, 'Habitability_Class'] = 1 - df.loc[noise_indices, 'Habitability_Class']
print(f"Introduced noise to {n_noise} samples.")

# 3. Separate Features (X) and Target (y)
X = df.drop(['Habitability_Class', 'Habitability_Index'], axis=1)
y = df['Habitability_Class']

# 4. Split: 80% to train the AI, 20% to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 5. Save the 4 pieces for Module 4
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print(f"Success! Created training set with {len(X_train)} planets.")