import csv
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



from collections import Counter

import joblib


lis_macd_lines = []

source_data_file = './ml_data/contract_data.csv'
filtered_data_file = 'base_training_data.csv'
training_data_file = 'ai_training_data.csv'
training_sequences = './training_sequences.npy'
training_labels = './training_labels.npy'
knn_model_save_path = '../ai-models/knn_model.pkl'
rf_model_save_path = '../ai-models/rf_model.pkl'
xgb_model_save_path = '../ai-models/xgb_model.pkl'


sequence_length = 10  # Number of rows in each sequence

def prepare_ai_training_data():

    # Mapping for 'B/S' values
    bs_mapping = {'BUY': 0, 'SELL': 1}

    if os.path.isfile(training_data_file):
        # Delete the old training data file
        os.remove(training_data_file)
        print(f"Deleted old training data: {training_data_file}")

    # Check if the source data file exists
    if not os.path.isfile(filtered_data_file):
        print(f"Source data file not found: {filtered_data_file}")
        return

    # Open the newly created datafile
    with open(filtered_data_file, mode="r", newline="") as source_file:
        reader = csv.reader(source_file)
        headers = next(reader)  # Read the headers
        rows = list(reader)  # Read all rows into a list

        training_rows = []  # Store modified rows

        for row in rows:
            if len(row) < 8:  # Ensure row has enough columns
                print(f"⚠️ Skipping incomplete row: {row}")
                continue

            row[7] = bs_mapping.get(row[7], row[7])  # Replace column index 10 with mapped value

            training_rows.append(row)  # Append modified row


        # Write to the new training data file
        with open(training_data_file, mode='w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(headers)  # Write headers
            writer.writerows(training_rows)  # Write rows
    print(f"New AI training data prepared and saved to {training_data_file}")


def creating_the_sequences():
    # Parameters
    features_columns = slice(0, -1)  # All columns except the last one for features
    label_column = -1  # Last column for the label

    # Load the data
    sequences = []
    labels = []
    data = []

    with open(training_data_file, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Skip headers
        for i, row in enumerate(reader):
            float_row = list(map(float, row))  # Attempt conversion to float
            data.append(float_row)

    # Convert the list of rows into a NumPy array
    data = np.array(data)
    # print(data)

    # Create sequences
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length, features_columns])
        labels.append(data[i + sequence_length - 1, -1])  # Label from the last column in the sequence

    # Convert to NumPy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)

    # Save for training
    np.save(training_sequences, sequences)
    np.save(training_labels, labels)

    # print(f"Saved {len(sequences)} sequences and {len(labels)} labels.")
    # print(f"Unique labels: {np.unique(labels)}")  # Should output [1, 2, 3]
    # print("Sample sequence with label:")
    # print(sequences[0])  # Features for the first sequence
    # print(labels[0])  # Corresponding label

    return


# Custom focal loss function (optional)
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * K.pow((1 - p_t), gamma) * bce
        return K.mean(loss)
    return loss


def ai_rnn_model_training():
    """Train an RNN model to classify BUY, SELL, HOLD based on MACD and candlestick patterns."""

    # Load preprocessed training data
    training_sequences = "./training_sequences.npy"
    training_labels = "./training_labels.npy"
    rnn_model_save_path = "./models/rnn_model.h5"

    sequences = np.load(training_sequences)
    labels = np.load(training_labels)

    # Check initial shape
    print(f"Loaded sequences shape: {sequences.shape}")  # (samples, timesteps, features)
    print(f"Loaded labels shape: {labels.shape}")  # (samples,)

    # Train-test split
    scaler = StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    # Oversampling (SMOTE) to balance classes
    smote = SMOTE(random_state=42, k_neighbors=1)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_flat, y_train)

    # Reshape back to (samples, timesteps, features)
    sequence_length = 7  # Or another value that fits
    num_features = 8
    total_samples = x_train_balanced.shape[0] // (sequence_length * num_features)
    x_train = x_train_balanced[:total_samples * sequence_length * num_features]  # Trim excess
    x_train = x_train.reshape(-1, sequence_length, num_features)

    # Convert labels to categorical (for softmax output)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Define the RNN model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, 8)),
        Dropout(0.35),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])

    lr = 0.0001  # Lower learning rate for stability
    optimizer = optimizer = RMSprop(learning_rate=lr)

    # Compile the model with focal loss or sparse categorical crossentropy
    model.compile(optimizer=optimizer, loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy'])

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Predictions
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Print evaluation metrics
    precision = precision_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_labels, y_pred, average='weighted')
    f1 = f1_score(y_test_labels, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test_labels, y_pred)
    print(y_test.shape, y_pred.shape)
    y_test = np.argmax(y_test, axis=1)


    print("RNN Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    print("R-NN Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    if not os.path.exists("./models"):
        os.makedirs("./models")

    model.save(rnn_model_save_path)
    print("✅ Model saved successfully!")

def ai_knn_model_training():
    # Load the training data
    data = pd.read_csv(training_data_file)

    distance_metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']

    # Separate features and labels
    features = data.iloc[:, :-1]  # All columns except the last one
    labels = data.iloc[:, -1]  # The last column contains 0,1,2

    # Scale the features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Train-test split for k-NN
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(features_scaled, labels, test_size=0.2,
                                                                        random_state=42)

    k_values = range(1, 21)
    cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_knn, y_train_knn, cv=5).mean() for k in
                 k_values]
    optimal_k = k_values[np.argmax(cv_scores)]
    print(f"Optimal k: {optimal_k}")
    #
    # import matplotlib.pyplot as plt
    # #
    # plt.plot(k_values, cv_scores, marker='o')
    # plt.xlabel('Number of Neighbors (k)')
    # plt.ylabel('Cross-Validation Accuracy')
    # plt.title('Optimal k Selection for k-NN')
    # plt.show()

    best_metric = None
    best_accuracy = 0

    # Train the k-NN model
    for metric in distance_metrics:
        knn = KNeighborsClassifier(n_neighbors=1, metric=metric)
        knn.fit(X_train_knn, y_train_knn)
        predictions = knn.predict(X_test_knn)
        accuracy = accuracy_score(y_test_knn, predictions)

        print(f"Metric: {metric}, Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_metric = metric

    print(f"\nBest distance metric: {best_metric} with accuracy {best_accuracy:.4f}")
    print(f"k-NN Accuracy: {best_accuracy:.4f}")
    print("KNN Classification Report:")
    #print(classification_report(y_test_knn, knn_predictions))

    # Save the trained model
    joblib.dump(knn, knn_model_save_path)
    # print(f"k-NN model saved to {knn_model_save_path}")

def models_training():
    # 🔹 Load the training data
    data = pd.read_csv(training_data_file)

    # 🔹 Separate features and labels
    features = data.iloc[:, :-1]  # All columns except the last one
    labels = data.iloc[:, -1]  # Last column (BUY/SELL)

    # 🔹 Scale the features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # 🔹 Save the scaler for later use in predictions
    joblib.dump(scaler, "scaler.pkl")

    # 🔹 Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    ### ✅ Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    ### ✅ Train XGBoost
    xgb_model = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    # 🔹 Evaluate both models
    print("\n📊 Random Forest Results:")
    print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
    print(classification_report(y_test, rf_predictions))

    print("\n📊 XGBoost Results:")
    print(f"Accuracy: {accuracy_score(y_test, xgb_predictions):.4f}")
    print(classification_report(y_test, xgb_predictions))

    # ✅ Save the models
    joblib.dump(rf_model, rf_model_save_path)
    joblib.dump(xgb_model, xgb_model_save_path)

    print("✅ Models saved successfully!")


def final_trading_decision(knn_decision, rf_decision, xgb_decision, bot_advice):
    """Determine final decision based on majority voting of the three models and bot advice."""

    decisions = [knn_decision, rf_decision, xgb_decision, bot_advice]
    final_decision = max(set(decisions), key=decisions.count)  # Majority voting

    return final_decision



    # # Determine final decision based on the majority
    # if buy_votes >= 2:
    #     return "Buy"
    # elif sell_votes >= 2:
    #     return "Sell"
    # elif hold_votes >= 2:
    #     return "Hold"
    # else:
    #     # If there is no clear majority, return the human advice as a default
    #     return advice


def make_a_prediction(input_values,advice):

    # Load the model
    """Make predictions using k-NN, Random Forest, and XGBoost, then determine a final decision."""

    # Convert input to NumPy array
    input_array = np.array(input_values, dtype=np.float32).reshape(1, -1)

    # Model Predictions
    knn_prediction = knn_model.predict(input_array)[0]
    rf_prediction = rf_model.predict(input_array)[0]
    xgb_prediction = xgb_model.predict(input_array)[0]

    # Convert numeric predictions to human-readable decisions
    decision_mapping = {0: "Buy", 1: "Sell"}

    knn_decision = decision_mapping.get(knn_prediction, "Hold")
    rf_decision = decision_mapping.get(rf_prediction, "Hold")
    xgb_decision = decision_mapping.get(xgb_prediction, "Hold")

    print(
        f"Bot suggests: {advice}, k-NN suggests: {knn_decision}, RF suggests: {rf_decision}, XGBoost suggests: {xgb_decision}")

    # Determine the final trading decision based on majority voting
    trading_advice = final_trading_decision(knn_decision, rf_decision, xgb_decision, advice)

    return trading_advice


def start_ai():
    # fetch_and_clean_data_rows()
    prepare_ai_training_data()
    creating_the_sequences()
    # ai_rnn_model_training()
    ai_knn_model_training()
    models_training()

if __name__ == "__main__":
     start_ai()

