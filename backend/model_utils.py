import json
import os
import re


# ---------------- NORMALIZATION ---------------- #

def normalize_column_name(col):
    import re
    import unicodedata

    # Normalize unicode (VERY IMPORTANT)
    col = unicodedata.normalize("NFKD", col)

    col = col.strip().lower()

    # Replace unicode subscripts (₂ → 2)
    col = col.replace("₂", "2")

    # Remove spaces, underscores, dots
    col = re.sub(r"[\s._]", "", col)

    # FORCE mapping
    mapping = {
        "pm25": "pm25",
        "pm2.5": "pm25",
        "pm2_5": "pm25",

        "pm10": "pm10",

        "no2": "no2",
        "so2": "so2",   
        "co": "co",
        "nh3": "nh3",

        "o3": "o3",
        "ozone": "o3",
    }

    return mapping.get(col, col)


def normalize_dataframe_columns(df):
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


# ---------------- METADATA ---------------- #

def save_model_metadata(model_name, features, timesteps, base_path=None):
    metadata = {
        "model_name": model_name,
        "features": [f.lower() for f in features],  # 🔥 ensure consistency
        "timesteps": timesteps,
        "num_features": len(features),
        "model_type": "deep_learning"
    }

    if base_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_path = os.path.join(project_root, "Models", model_name)

    os.makedirs(base_path, exist_ok=True)

    path = os.path.join(base_path, f"{model_name}_metadata.json")

    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)


def load_model_metadata(model_name, base_path=None):
    if base_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_path = os.path.join(project_root, "Models", model_name)

    path = os.path.join(base_path, f"{model_name}_metadata.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata not found at: {path}")

    with open(path, "r") as f:
        return json.load(f)


# ---------------- VALIDATION ---------------- #

def validate_features(df, required_features, strict=True):
    df = normalize_dataframe_columns(df)

    required_features = [col.lower() for col in required_features]
    df.columns = [col.lower() for col in df.columns]

    input_features = list(df.columns)

    missing = list(set(required_features) - set(input_features))
    extra = list(set(input_features) - set(required_features))

    if strict and missing:
        return df, missing, extra, False

    if strict:
        df = df.drop(columns=extra, errors="ignore")
    else:
        df = df.reindex(columns=required_features, fill_value=0)

    return df, missing, extra, True