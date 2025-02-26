"""
This script performs hyperparameter tuning for the Support Vector Machine (SVM) regressor
on the UCI Algerian Forest Fires dataset. It preprocesses the data using StandardScaler for
feature normalization and utilizes GridSearchCV to find the optimal hyperparameters for the SVM algorithm.

Jaskirat Kaur, 000904397
Date: February 22, 2025

"""
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
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

# Define the hyperparameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf'
    'epsilon': [0.1, 0.2, 0.5]  # Epsilon in the epsilon-SVR model
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(data_train, target_train)

# Output the best parameters and corresponding Mean Squared Error (MSE)
print("Best Parameters for SVM:", grid_search.best_params_)
print(f"Best MSE: {-grid_search.best_score_:.4f}\n")
