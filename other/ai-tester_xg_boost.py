import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product

# Load training data
data = pd.read_csv("../ml_data/ai_training_data.csv")  # Pas het pad aan naar jouw dataset
features = data.iloc[:, :-1]  # Alle kolommen behalve de laatste
labels = data.iloc[:, -1]  # Laatste kolom (labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define hyperparameter grids
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

# Create CSV file for results
csv_filename = "hyperparameter_results_rf_xgb.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "n_estimators", "max_depth", "min_samples_split", "learning_rate",
                     "Precision", "Recall", "F1-Score", "Test Accuracy"])

# Function to evaluate and save results
def evaluate_model(model, model_name, params, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)

    # Save results to CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([model_name, *params, precision, recall, f1, accuracy])

    print(f"✅ Finished training {model_name} with {params} | F1-Score: {f1:.4f}")

# Loop through all hyperparameter combinations for Random Forest
for params in product(*rf_param_grid.values()):
    n_estimators, max_depth, min_samples_split = params

    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Evaluate and save results
    evaluate_model(rf_model, "Random Forest", params, X_test, y_test)

# Loop through all hyperparameter combinations for XGBoost
for params in product(*xgb_param_grid.values()):
    n_estimators, max_depth, learning_rate = params

    # Train XGBoost
    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate and save results
    evaluate_model(xgb_model, "XGBoost", params, X_test, y_test)

print(f"🔍 All results saved to {csv_filename}")