import numpy as np
import pandas as pd
from Preprocessing.aqi_calculator import compute_aqi
from backend.model_manager import load_selected_model

from backend.event_handlers.statistical import StatisticalEventHandler

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def validate_sequence_input(input_df, metadata):
    required_columns = [col.lower() for col in metadata["features"]]
    input_columns = [col.lower() for col in input_df.columns]

    missing = set(required_columns) - set(input_columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def predict_with_model(model_name, input_df, event_type=None):

    model, scaler, metadata = load_selected_model(model_name)
    model_type = metadata.get("model_type")

    if model_type == "deep_learning":

        validate_sequence_input(input_df, metadata)

        # 🔥 Use metadata
        feature_cols = [col.lower() for col in metadata["features"]]
        timesteps = metadata.get("timesteps", 24)

        input_df.columns = [col.lower() for col in input_df.columns]

        # 🔥 Safety check
        if len(input_df) < timesteps:
            raise ValueError(f"Need at least {timesteps} rows for prediction")

        input_seq = input_df.tail(timesteps)[feature_cols]

        # Scale
        scaled_input = scaler.transform(input_seq)
        X = np.expand_dims(scaled_input, axis=0)

        # Predict
        pred_scaled = model.predict(X)
        pred_pollutants = scaler.inverse_transform(pred_scaled)

        # 🔥 Use statistical event handler
        if event_type:
            handler = StatisticalEventHandler()
            pred_pollutants = handler.apply(
                pred_pollutants,
                feature_cols,
                event_type
            )

        # Ensure no negatives
        pred_pollutants = np.maximum(pred_pollutants, 0)

        pred_df = pd.DataFrame(pred_pollutants, columns=feature_cols)

        # 🔥 AQI computation (already normalized)
        predicted_aqi = compute_aqi(pred_df.iloc[0])
        category = get_aqi_category(predicted_aqi)

        print("Predicted pollutants:")
        print(pred_df.iloc[0])

        return {
            "type": "deep_learning",
            "predicted_pollutants": pred_df.iloc[0].to_dict(),
            "AQI": float(predicted_aqi),
            "Category": category
        }

    elif model_type == "regression":

        X = input_df.select_dtypes(include=["number"])

        if X.empty:
            raise ValueError("No numeric features found for regression model.")

        prediction = model.predict(X.iloc[-1:].values)[0]
        prediction = max(float(prediction), 0)

        category = get_aqi_category(prediction)

        return {
            "type": "regression",
            "predicted_pollutants": None,
            "AQI": prediction,
            "Category": category
        }

    else:
        raise ValueError("Unsupported model type.")