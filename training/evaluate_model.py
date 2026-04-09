import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from Preprocessing.aqi_calculator import compute_aqi


df = pd.read_csv("../Datasets/Cleaned Datasets/cleaned_hourly_data.csv")


actual_aqi = df["AQI"].values


df_pollutants = df.drop(columns=["timestamp", "AQI"])


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_pollutants)
scaled_df = pd.DataFrame(scaled_data, columns=df_pollutants.columns)




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

X_test = X[split:]
y_test = y[split:]


model = load_model("./model.keras") 


pred_scaled = model.predict(X_test)


pred_pollutants = scaler.inverse_transform(pred_scaled)


pred_pollutants = np.maximum(pred_pollutants, 0)


pred_df = pd.DataFrame(pred_pollutants, columns=df_pollutants.columns)

pred_df["AQI_pred"] = pred_df.apply(compute_aqi, axis=1)


actual_aqi_test = actual_aqi[window_size + split:]


valid_mask = (~pred_df["AQI_pred"].isna()) & (~np.isnan(actual_aqi_test))

pred_df = pred_df[valid_mask]
actual_aqi_test = actual_aqi_test[valid_mask]


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(actual_aqi_test, pred_df["AQI_pred"])
rmse = np.sqrt(mean_squared_error(actual_aqi_test, pred_df["AQI_pred"]))

print("AQI MAE:", mae)
print("AQI RMSE:", rmse)

results_df = pd.DataFrame({
    "Actual_AQI": actual_aqi_test,
    "Predicted_AQI": pred_df["AQI_pred"].values
})

results_df.to_csv("aqi_predictions.csv", index=False)

print("Saved predictions to aqi_predictions.csv")


import matplotlib.pyplot as plt

plt.figure()

plt.plot(results_df["Actual_AQI"].values, label="Actual AQI")
plt.plot(results_df["Predicted_AQI"].values, label="Predicted AQI")

plt.xlabel("Time Steps")
plt.ylabel("AQI")
plt.title("Actual vs Predicted AQI")
plt.legend()

plt.savefig("aqi_comparison.png")  # Save image for paper
plt.show()