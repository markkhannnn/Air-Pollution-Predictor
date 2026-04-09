import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Models.cnn_lstm_model import build_cnn_lstm
from backend.model_utils import save_model_metadata


df = pd.read_csv("../Datasets/Cleaned Datasets/cleaned_hourly_data.csv")



df = df.drop(columns=["timestamp", "aqi"], errors="ignore")

df = df.select_dtypes(include=["number"])
from backend.model_utils import normalize_dataframe_columns
print("RAW columns:", df.columns.tolist())
df = normalize_dataframe_columns(df)
print("NORMALIZED columns:", df.columns.tolist())

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)


def create_sequences(data, window_size=24):
    X = []
    y = []
    
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data.iloc[i+window_size].values)
    
    return np.array(X), np.array(y)

window_size = 24
X, y = create_sequences(scaled_df, window_size)


split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


model = build_cnn_lstm(
    input_shape=(X.shape[1], X.shape[2]),
    output_dim=y.shape[1]
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
base_path = "../Models/CNN_LSTM_DEFAULT"
os.makedirs(base_path, exist_ok=True)
model.save(f"{base_path}/model.keras")

# Save metadata
model_name = "CNN_LSTM_DEFAULT"
features = list(df.columns)
timesteps = window_size

save_model_metadata(
    model_name=model_name,
    features=features,
    timesteps=timesteps,
    base_path=base_path
)

print("Metadata saved successfully.")

import joblib

joblib.dump(scaler, f"{base_path}/scaler.pkl")

print("Scaler saved successfully.")

print("Model training complete and saved.")