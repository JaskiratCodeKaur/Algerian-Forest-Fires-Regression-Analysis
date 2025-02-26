"""
This script performs hyperparameter tuning for a Decision Tree Regressor
on the UCI Algerian Forest Fires dataset. It preprocesses the data using StandardScaler for
feature normalization and utilizes GridSearchCV to find the optimal hyperparameters for the Decision Tree model.

Jaskirat Kaur, 000904397
Date: February 22, 2025

"""

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# Define the hyperparameter grid for the Decision Tree Regressor
param_grid = {
    'criterion': ['squared_error', 'friedman_mse'],  # Splitting criterion
    'max_depth': [None, 5, 10, 15, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples required at a leaf node
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(data_train, target_train)

# Output the best parameters and corresponding Mean Squared Error (MSE)
print("Best Parameters for Decision Tree:", grid_search.best_params_)
print(f"Best MSE: {-grid_search.best_score_:.4f}\n")
