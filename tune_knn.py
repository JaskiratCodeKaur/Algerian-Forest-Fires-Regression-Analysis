"""
This script performs hyperparameter tuning for the K-Nearest Neighbors (K-NN) regressor
on the UCI Algerian Forest Fires dataset. It preprocesses the data using StandardScaler for
feature normalization and utilizes GridSearchCV to find the optimal hyperparameters for the K-NN method.

Jaskirat Kaur, 000904397
Date: 22nd February, 2025

"""

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Load dataset from CSV (skip header row)
raw = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1, dtype=str)

# Separate numerical features (all columns except the last) and convert them to float
data = raw[:, :-1].astype(float) 

# Extract the target variable (last column) which is categorical (string)
target = raw[:, -1]

# Encode categorical labels: 'fire' -> 1.0, others -> 0.0
target = np.where(raw[:, -1] == 'fire', 1, 0).astype(float)

# Shuffle data
data, target = shuffle(data, target, random_state=42)

# Split into training (80%) and testing (20%) sets
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# Define parameter grid for K-NN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Try different values for k
    'weights': ['uniform', 'distance'],  # Weighting strategies
    'p': [1, 2]  # Manhattan (p=1) and Euclidean (p=2) distance metrics
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(data_train, target_train)

# Output best parameters and corresponding MSE
print("Best Parameters for K-NN:", grid_search.best_params_)
print(f"Best MSE: {-grid_search.best_score_:.4f}\n")
