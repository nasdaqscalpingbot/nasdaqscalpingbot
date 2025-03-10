import csv
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score

from imblearn.over_sampling import SMOTE

from collections import Counter

import joblib


lis_macd_lines = []

source_data_file = './ml_data/contract_data.csv'
filtered_data_file = './base_training_data.csv'
training_data_file = './ai_training_data.csv'
training_sequences = './training_sequences.npy'
training_labels = './training_labels.npy'
rnn_model_save_path = './models/lstm_model.keras'
knn_model_save_path = './models/knn_model.pkl'

sequence_length = 10  # Number of rows in each sequence

# def fetch_and_clean_data_rows():
#
#     filtered_rows = []
#
#     columns_to_keep = [
#         'Current candle open', 'Current candle close', 'Current candle high', 'Current candle low',
#         'Previous candle open', 'Previous candle close', 'Previous candle high', 'Previous candle low',
#         'Oldest candle open', 'Oldest candle close', 'Oldest candle high', 'Oldest candle low',
#         'Current candle netchange', 'Previous candle netchange', 'Oldest candle netchange',
#         'avg_change', 'percentageChange', 'spread', 'std dev',
#         'MACD line', 'Signal_line', 'Histogram', 'Status', 'Buy/Sell'
#     ]
#
#     valid_values = {2, 1, 0}  # Valid values for 'B/S' column
#
#
#     if not os.path.isfile(source_data_file):
#         print(f"Source data file not found: {source_data_file}")
#         return
#
#     if os.path.isfile(filtered_data_file):
#         # Delete the old training data file
#         os.remove(filtered_data_file)
#         print(f"Deleted old training data: {filtered_data_file}")
#
#     # Read rows from the existing source CSV
#     with open(source_data_file, mode='r') as source_file:
#         reader = csv.reader(source_file)
#         headers = next(reader)  # Read the headers
#         # Map header indices
#         header_indices = {header: idx for idx, header in enumerate(headers)}
#         # Find indices of columns to keep
#         indices_to_keep = [header_indices[col] for col in columns_to_keep if col in header_indices]
#         # Index of 'B/S' column
#         bs_index = header_indices['Buy/Sell']
#         # Filter the rows
#         for row in reader:
#             if row:  # Skip empty rows
#                 if row[bs_index] in valid_values and "N/A" not in [row[idx] for idx in indices_to_keep]:
#                     filtered_row = [row[idx] for idx in indices_to_keep]
#                     filtered_rows.append(filtered_row)
#
#     # If needed, write filtered data back to a new CSV
#     with open(filtered_data_file, mode='w', newline='') as output_file:
#         writer = csv.writer(output_file)
#         # Write headers
#         writer.writerow([headers[idx] for idx in indices_to_keep])
#         # Write filtered rows
#         writer.writerows(filtered_rows)



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

        # Modify headers to reflect changes
        trimmed_headers = headers[:8] + ["B/S"]  # Keep first 8 columns and replace column 8

        training_rows = []  # Store modified rows

        for row in rows:
            if len(row) < 11:  # Ensure row has enough columns
                print(f"⚠️ Skipping incomplete row: {row}")
                continue

            row[8] = bs_mapping.get(row[8], row[8])  # Replace column index 8 with mapped value
            row[10] = bs_mapping.get(row[10], row[10])  # Replace column index 10 with mapped value

            training_rows.append(row)  # Append modified row


        # Write to the new training data file
        with open(training_data_file, mode='w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(trimmed_headers)  # Write headers
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


# Focal loss is designed for imbalanced datasets. It modifies the standard loss to focus
# more on hard-to-classify examples:
# def focal_loss(alpha, gamma):
#     def loss_fn(y_true, y_pred):
#         y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
#         cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
#         loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
#         return tf.reduce_sum(loss, axis=-1)
#     return loss_fn



# Function to test different class weights
# def tune_class_weights(model):
#     sequences = np.load(training_sequences)
#     labels = np.load(training_labels)
#     x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
#
#     hold_start = 0.5
#     hold_end = 1.0
#     hold_step = 0.00001
#
#     buy_start = 2.2
#     buy_end = 2.8
#     buy_step = 0.0001
#
#     sell_start = 1.5
#     sell_end = 2.0
#     sell_step = 0.1
#
#
#     best_f1 = 0
#     best_weights = None
#     hold_weight = hold_start
#     while hold_weight <= hold_end:
#         buy_weight = buy_start
#         while buy_weight <= buy_end:
#             sell_weight = sell_start
#             while sell_weight <= sell_end:
#                 print("Nieuwe waarden: ",hold_weight, buy_weight, sell_weight)
#                 class_weights_dict = {0: hold_weight, 1: buy_weight, 2: sell_weight}
#                 model.fit(x_train, y_train, class_weight=class_weights_dict, epochs=10, batch_size=32)
#                 y_pred = model.predict(x_test).argmax(axis=1)
#                 f1 = f1_score(y_test, y_pred, average="weighted")
#                 if f1 > best_f1:
#                     best_f1 = f1
#                     best_weights = class_weights_dict
#                 # Increment the sell weight
#                 sell_weight += sell_step
#             # Increment the buy weight
#             buy_weight += buy_step
#         # Increment the hold weight
#         hold_weight += hold_step
#     print("Best F1-Score:", best_f1)
#     print("Best Class Weights:", best_weights)
#     return best_weights





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
    x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    # Oversampling (SMOTE) to balance classes
    smote = SMOTE(random_state=42, k_neighbors=1)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_flat, y_train)

    # Reshape back to (samples, timesteps, features)
    sequence_length = 10  # Modify if needed
    x_train = x_train_balanced.reshape(-1, sequence_length, 10)
    y_train = y_train_balanced

    # Convert labels to categorical (for softmax output)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # Define the RNN model
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(sequence_length, 10)),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')  # 2 output classes: BUY, SELL
    ])

    # Compile the model with focal loss or sparse categorical crossentropy
    model.compile(optimizer='adam', loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy'])

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
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

    print("RNN Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

    # Save model
    if not os.path.exists("./models"):
        os.makedirs("./models")

    model.save(rnn_model_save_path)
    print("✅ Model saved successfully!")

def ai_knn_model_training():
    # Load the training data
    data = pd.read_csv(training_data_file)

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
    # print(f"Optimal k: {optimal_k}")
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(k_values, cv_scores, marker='o')
    # plt.xlabel('Number of Neighbors (k)')
    # plt.ylabel('Cross-Validation Accuracy')
    # plt.title('Optimal k Selection for k-NN')
    # plt.show()

    # Train the k-NN model
    k = 9  # Number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train_knn)
    # print("k-NN trained successfully!")

    # Evaluate k-NN
    knn_predictions = knn.predict(X_test_knn)
    knn_accuracy = accuracy_score(y_test_knn, knn_predictions)
    print(f"k-NN Accuracy: {knn_accuracy:.4f}")

    # Save the trained model
    joblib.dump(knn, knn_model_save_path)
    # print(f"k-NN model saved to {knn_model_save_path}")


def final_trading_decision(rnn_decision, knn_decision, advice):
    # Count votes for each decision
    decisions = [rnn_decision, knn_decision, advice]
    buy_votes = decisions.count("Buy")
    sell_votes = decisions.count("Sell")
    hold_votes = decisions.count("Hold")

    # alternative after testing
    # decision_scores = {
    #     "Buy": weights[0] * (rnn_decision == "Buy") + weights[1] * (knn_decision == "Buy") + weights[2] * (advice == "Buy"),
    #     "Sell": weights[0] * (rnn_decision == "Sell") + weights[1] * (knn_decision == "Sell") + weights[2] * (advice == "Sell"),
    #     "Hold": weights[0] * (rnn_decision == "Hold") + weights[1] * (knn_decision == "Hold") + weights[2] * (advice == "Hold"),
    # }



    # Determine final decision based on the majority
    if buy_votes >= 2:
        return "Buy"
    elif sell_votes >= 2:
        return "Sell"
    elif hold_votes >= 2:
        return "Hold"
    else:
        # If there is no clear majority, return the human advice as a default
        return advice


def make_a_prediction(input_values,advice):

    # Load the model
    rnn_model = load_model(rnn_model_save_path)
    knn_model = joblib.load(knn_model_save_path)
    print("Both models loaded successfully!")

    # Preprocess the input values
    # Assuming the input values are normalized the same way as training data
    input_array = np.array(input_values, dtype=np.float32).reshape((1, 1, 18))

    # RNN Prediction
    rnn_prediction = rnn_model.predict(input_array)
    rnn_decision = "Hold"  # Default decision

    # Get the prediction
    # bs_mapping = 'BUY': 1, 'HOLD': 0, 'SELL': 2
    # Map the RNN output to the correct decision based on your bs_mapping
    rnn_output = np.round(rnn_prediction[0][0])  # Round to nearest integer
    if rnn_output == 0:
        rnn_decision = "Buy"
    elif rnn_output == 1:
        rnn_decision = "Sell"

    # Preprocess the input values for k-NN
    knn_input_array = np.array(input_values, dtype=np.float32).reshape(1, -1)
    knn_prediction = knn_model.predict(knn_input_array)[0]

    # Map the k-NN output to the correct decision based on your bs_mapping
    knn_decision = "Hold"  # Default decision
    if knn_prediction == 0:
        knn_decision = "Buy"
    elif knn_prediction == 1:
        knn_decision = "Sell"

    print(f"Bot suggests: {advice}", f"KNN suggests: {knn_decision}", f"RNN suggests: {rnn_decision}")

    trading_advice = final_trading_decision(rnn_decision, knn_decision, advice)
    return trading_advice



def start_ai():
    # fetch_and_clean_data_rows()
    prepare_ai_training_data()
    creating_the_sequences()
    ai_rnn_model_training()
    # ai_knn_model_training()

if __name__ == "__main__":
     start_ai()

# Precision: 0.15630070308274743
# Recall: 0.3953488372093023
# F1-Score: 0.22403100775193796
# Confusion Matrix:
#  [[ 0 11  0]
#  [ 0 34  0]
#  [ 0 41  0]]
# k-NN Accuracy: 0.7955
# k-NN model saved to ./models/knn_model.pkl
# The bot has started making profit


