import itertools
import csv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load training data
sequences = np.load("../ml_data/training_sequences.npy")
labels = np.load("../ml_data/training_labels.npy")

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Apply SMOTE (oversampling)
smote = SMOTE(random_state=42, k_neighbors=1)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train_flat, y_train)
x_train = x_train_balanced.reshape(-1, 10, 9)  # Reshape back to (samples, timesteps, features)
y_train = to_categorical(y_train_balanced, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Define hyperparameter grid
hyperparameter_grid = {
    "batch_size": [32, 64, 128],
    "epochs": [50, 70],
    "dropout": [0.3, 0.35, 0.4],
    "learning_rate": [0.0001, 0.0003, 0.0005],
    "lstm_units_1": [128, 256],
    "lstm_units_2": [64, 128],
    "dense_units": [32, 64],
    "optimizer": ["adam", "rmsprop"]
}

# Create CSV file for results
csv_filename = "hyperparameter_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Batch Size", "Epochs", "Dropout", "LR", "LSTM1", "LSTM2", "Dense", "Optimizer",
                     "Precision", "Recall", "F1-Score", "Test Accuracy"])

# Loop through all hyperparameter combinations
for params in itertools.product(*hyperparameter_grid.values()):
    batch_size, epochs, dropout, lr, lstm1, lstm2, dense, opt = params

    # Define optimizer
    if opt == "adam":
        optimizer = Adam(learning_rate=lr)
    elif opt == "rmsprop":
        optimizer = RMSprop(learning_rate=lr)
    else:
        optimizer = SGD(learning_rate=lr)

    # Define model
    model = Sequential([
        Input(shape=(10, 9)),  # Explicit input shape
        LSTM(lstm1, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm2, return_sequences=False),
        Dropout(dropout),
        Dense(dense, activation="relu"),
        Dense(2, activation="softmax")
    ])

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    precision = precision_score(y_test_labels, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test_labels, y_pred, average="weighted")
    f1 = f1_score(y_test_labels, y_pred, average="weighted")

    # Save results to CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([batch_size, epochs, dropout, lr, lstm1, lstm2, dense, opt, precision, recall, f1, accuracy])

    print(f"✅ Finished training with {params} | F1-Score: {f1:.4f}")

print(f"🔍 All results saved to {csv_filename}")
