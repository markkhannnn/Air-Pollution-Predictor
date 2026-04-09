import os
import json
import joblib
from tensorflow.keras.models import load_model # type: ignore


MODELS_DIR = "Models"


def list_available_models():
    """
    Returns only valid model folders (ignores __pycache__ etc.)
    """
    models = []

    for folder in os.listdir(MODELS_DIR):

        
        if folder.startswith("__"):
            continue

        path = os.path.join(MODELS_DIR, folder)

        if not os.path.isdir(path):
            continue

        # ✅ Check if valid model folder
        model_file = os.path.join(path, "model.keras")
        metadata_file = os.path.join(path, f"{folder}_metadata.json")

        if os.path.exists(model_file) and os.path.exists(metadata_file):
            models.append(folder)

    return models

def load_selected_model(model_name):
    """
    Loads model and metadata dynamically based on folder.
    Returns:
        model
        scaler (if exists, else None)
        metadata
    """
    model_path = os.path.join(MODELS_DIR, model_name)

    metadata_path = os.path.join(model_path, f"{model_name}_metadata.json")

    print(f"Looking for metadata at: {metadata_path}")

    if not os.path.exists(metadata_path):
        raise ValueError(f"No metadata found at {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model_type = metadata.get("model_type")

    if model_type == "deep_learning":
        model_file = os.path.join(model_path, "model.keras")
        scaler_file = os.path.join(model_path, "scaler.pkl")

        model = load_model(model_file)

        if not os.path.exists(scaler_file):
            raise ValueError("Scaler file missing for deep learning model.")

        scaler = joblib.load(scaler_file)

        return model, scaler, metadata

    elif model_type == "regression":
        model_file = os.path.join(model_path, "model.pkl")

        if not os.path.exists(model_file):
            raise ValueError("Regression model file not found.")

        model = joblib.load(model_file)

        return model, None, metadata

    else:
        raise ValueError(f"Unsupported model type: {model_type}")