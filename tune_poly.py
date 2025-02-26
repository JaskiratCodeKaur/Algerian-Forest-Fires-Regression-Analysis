"""
This script performs hyperparameter tuning for a Polynomial Regression model
on the UCI Algerian Forest Fires dataset. It constructs the machine learning
pipeline to preprocess the data and utilizes GridSearchCV to identify the
optimal polynomial degree for the regression model.

Jaskirat Kaur, 000904397
Date: February 22, 2025

"""

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
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

# Create a pipeline that includes PolynomialFeatures and LinearRegression
pipeline = Pipeline(steps=[
    ('poly', PolynomialFeatures()),  # Apply polynomial features
    ('regressor', LinearRegression())  # Apply linear regression
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'poly__degree': [2,3,4,5]  # Test polynomial degrees from 1 to 5
}

# Perform GridSearchCV to find the best polynomial degree
grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(data_train, target_train)

# Output the best parameters and corresponding Mean Squared Error (MSE)
print("Best Parameters for Polynomial Regression:", grid_search.best_params_)
print(f"Best MSE: {-grid_search.best_score_:.4f}\n")  # Negate the negative MSE to display the positive value