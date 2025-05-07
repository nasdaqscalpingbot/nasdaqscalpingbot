import csv
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# import ai_logging
import joblib

source_data_file = './ml_data/contract_data.csv'
filtered_data_file = './ml_data/base_training_data.csv'
training_data_file = './ml_data/ai_training_data.csv'
training_sequences = './training_sequences.npy'
training_labels = './training_labels.npy'
knn_model_save_path = './ai-models/knn_model.pkl'
rf_model_save_path = './ai-models/random_forest_model.pkl'
xgb_model_save_path = './ai-models/xgboost_model.pkl'


sequence_length = 10  # Number of rows in each sequence
def refine_hold_labels(df):
    hold_mask = (
        (df['ADX'] < 15) &                      # Stronger trend filter
        (abs(df['MACD'] - df['Signal']) < 0.5) &  # Neutral momentum
        (df['ATR'] > df['ATR'].rolling(50).mean() * 0.7)  # Active volatility
    )
    df.loc[hold_mask, 'Label'] = 'HOLD'
    return df


def train_all_models():
    # 🔹 Load data
    data = refine_hold_labels(pd.read_csv(training_data_file))

    # 🔹 Convert string labels to numerical (critical fix)
    label_encoder = LabelEncoder()
    features = data.iloc[:, :-1]
    print("Aantal features in training dataset:", features.shape[1])
    labels = label_encoder.fit_transform(data.iloc[:, -1])  # BUY=0, SELL=1, HOLD=2

    # 🔹 Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    print("Schaalvorm:", features_scaled.shape)  # Moet (aantal rijen, 18) zijn

    joblib.dump(scaler, "ml_data/scaler.pkl")
    joblib.dump(label_encoder, "ml_data/label_encoder.pkl")  # Save for live trading

    # 🔹 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # class_weights = {0: 1, 1: 1, 2: 3}  # Triple penalty for misclassifying HOLD

    # 🔹 Models (now handling 3 classes)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.2,
            random_state=42,
            objective='multi:softmax',
        )
    }

    # 🔹 Train and evaluate
    for name, model in models.items():
        print(f"\n🔥 Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(f"{name} Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions,
                                    target_names=label_encoder.classes_))  # Show BUY/SELL/HOLD
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        joblib.dump(model, f"./ai-models/{name.lower().replace(' ', '_')}_model.pkl")




def make_a_prediction(input_values, bot_advice):
    """Make predictions using Random Forest and XGBoost, then determine a final decision with confidence."""

    # Load models
    rf_model = joblib.load(rf_model_save_path)
    xgb_model = joblib.load(xgb_model_save_path)

    # Convert input to NumPy array
    input_array = np.array(input_values, dtype=np.float32).reshape(1, -1)

    # Model Predictions
    rf_prediction = rf_model.predict(input_array)[0]
    xgb_prediction = xgb_model.predict(input_array)[0]

    # Get confidence scores using predict_proba()
    rf_probabilities = rf_model.predict_proba(input_array)
    xgb_probabilities = xgb_model.predict_proba(input_array)

    rf_confidence = max(rf_probabilities[0])  # Confidence of RF decision
    xgb_confidence = max(xgb_probabilities[0])  # Confidence of XGB decision

    rf_class_probs = dict(zip(rf_model.classes_, rf_probabilities[0]))
    xgb_class_probs = dict(zip(xgb_model.classes_, xgb_probabilities[0]))

    # print("Volledige RF proba:", rf_class_probs)
    # print("Volledige XGB proba:", xgb_class_probs)


    # Convert numeric predictions to human-readable decisions
    decision_mapping = {0: "BUY", 1: "SELL"}
    rf_decision = decision_mapping.get(rf_prediction, "HOLD")
    xgb_decision = decision_mapping.get(xgb_prediction, "HOLD")

    # print(
    #     f"Bot suggests: {bot_advice}, RF suggests: {rf_decision} (Confidence: {rf_confidence:.2f}), "
    #     f"XGBoost suggests: {xgb_decision} (Confidence: {xgb_confidence:.2f})"
    # )

    # Count model agreement
    predictions = [rf_decision, xgb_decision]
    most_common_decision = max(set(predictions), key=predictions.count)
    agreement_count = predictions.count(most_common_decision)

    # Confidence calculation (higher agreement = higher confidence)
    avg_confidence = (rf_confidence + xgb_confidence) / 2  # Average confidence

    # Determine the final trading decision based on majority voting
    decisions = [rf_decision, xgb_decision, bot_advice]
    trading_advice = max(set(decisions), key=decisions.count)  # Majority voting

    # hold_confidence = (rf_class_probs.get(2, 0) + xgb_class_probs.get(2, 0)) / 2
    # if hold_confidence > 0.4:
    #     trading_advice = "HOLD"

    # Sla AI-beslissingen op in de log-cache
    # ai_logging.store_ai_decision(rf_decision, rf_confidence, xgb_decision, xgb_confidence, trading_advice)

    return trading_advice, avg_confidence


def start_ai():
    train_all_models()

if __name__ == "__main__":
     start_ai()

