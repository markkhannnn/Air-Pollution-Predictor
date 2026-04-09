import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Preprocessing.aqi_calculator import compute_aqi


# ---------------- CONFIG ---------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "Datasets", "Cleaned Datasets", "cleaned_hourly_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Datasets", "with_events.csv")

SPIKE_THRESHOLD_PERCENTILE = 90  # top 10% spikes → events


# ---------------- STEP 1: LOAD ---------------- #
def load_data():
    df = pd.read_csv(DATA_PATH)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    return df


# ---------------- STEP 2: COMPUTE AQI ---------------- #
def add_aqi(df):
    print("Computing AQI...")

    df["AQI"] = df.apply(lambda row: compute_aqi(row), axis=1)

    return df


# ---------------- STEP 3: COMPUTE CHANGE ---------------- #
def compute_aqi_change(df):
    print("Computing AQI change...")

    df["aqi_change"] = df["AQI"].diff()

    return df


# ---------------- STEP 4: DETECT SPIKES ---------------- #
def detect_spikes(df):
    print("Detecting spikes...")

    threshold = df["aqi_change"].quantile(SPIKE_THRESHOLD_PERCENTILE / 100)

    df["is_spike"] = df["aqi_change"] > threshold

    print(f"Spike threshold: {threshold:.2f}")

    return df, threshold


# ---------------- STEP 5: ASSIGN EVENTS ---------------- #
def assign_events(df):
    print("Assigning event labels...")

    df["event"] = "none"

    # ---------------- PRIORITY ORDER ---------------- #
    # industrial < traffic < festival

    # 1. Industrial → sustained high pollution (low variation)
    df.loc[
        (df["timestamp"].dt.weekday < 5) &
        (df["AQI"] > df["AQI"].quantile(0.6)) &
        (df["aqi_change"].abs() < df["aqi_change"].quantile(0.7)),
        "event"
    ] = "industrial"

    # 2. Traffic → moderate increase + rush hours
    df.loc[
    (df["event"] == "none") &
    (df["aqi_change"] > df["aqi_change"].quantile(0.75)) &
    (df["timestamp"].dt.hour.isin([8, 9, 18, 19])),
    "event"
] = "traffic"

    # 3. Festival → sharp spike (highest priority)
    df.loc[
        (df["event"] == "none") &
        (df["is_spike"]),
        "event"
    ] = "festival"

    return df


# ---------------- STEP 6: SAVE ---------------- #
def save(df):
    print("Saving dataset with events...")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to: {OUTPUT_PATH}")


# ---------------- MAIN ---------------- #
def main():
    df = load_data()

    df = add_aqi(df)

    df = compute_aqi_change(df)

    df, threshold = detect_spikes(df)

    df = assign_events(df)

    print("\nEvent distribution:")
    print(df["event"].value_counts())

    save(df)


if __name__ == "__main__":
    main()