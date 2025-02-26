"""
This script evaluates multiple regression models (Linear Regression, Polynomial Regression, K-NN, Decision Tree, and SVR) 
on the Algerian Forest Fires Dataset, which contains meteorological and geographical data to predict forest fire occurrences. 
It preprocesses the data by encoding categorical labels, normalizing features, and splitting into training/testing sets. 
The models are evaluated using 5-fold cross-validation, reporting average, minimum, and maximum Mean Squared Error (MSE) for each. 

The dataset is sourced from the UCI Machine Learning Repository and is used for predictive analysis on fire occurrence based on weather and environmental conditions.
Reference: https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset

Jaskirat Kaur, 000904397
Date: 22nd February, 2025

"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset from CSV (skip header row)
raw = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1, dtype=str)

# Separate numerical features (all columns except the last) and convert them to float
data = raw[:, :-1].astype(float) 

# Extract the target variable (last column) which is categorical (string)
target = raw[:, -1]

# Encode categorical labels: 'fire' -> 1.0, others -> 0.0
target = np.where(raw[:, -1] == 'fire', 1, 0).astype(float)

# Shuffle the data
data, target = shuffle(data, target, random_state=42)

# Split the data into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# Output the size of the training and testing sets
print("Training set size:", data_train.shape)
print("Testing set size:", data_test.shape)
print("Number of features:", data_train.shape[1])

# Define models to evaluate with best hyperparameters
models = {
    'Linear Regression': LinearRegression(),
    'Polynomial Regression (degree=2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'K-NN Regression': KNeighborsRegressor(n_neighbors=3, p=1, weights='distance'),  # Tuned K-NN
    'Decision Tree Regression': DecisionTreeRegressor(criterion='squared_error', max_depth=None, min_samples_leaf=1, min_samples_split=10, random_state=42),  # Tuned Decision Tree
    'Support Vector Regression': SVR(C=1, epsilon=0.1, gamma='scale', kernel='rbf')  # Tuned SVR
}

# Evaluate each model using cross-validation
print("\nModel Evaluation Results:")
for name, model in models.items():
    results = cross_validate(model, data_train, target_train, scoring="neg_mean_squared_error", cv=5)
    mse_scores = -results["test_score"]  # Convert to positive MSE values
    print(f"{name}:")
    print(f"  Average MSE: {mse_scores.mean():.4f}")
    print(f"  Minimum MSE: {mse_scores.min():.4f}")
    print(f"  Maximum MSE: {mse_scores.max():.4f}")
    print()
