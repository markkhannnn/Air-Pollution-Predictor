import pandas as pd
import json
import os

# ---------------- PATH SETUP ---------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "Datasets", "with_events.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "backend", "event_handlers", "event_impacts.json")

FEATURES = ["co", "nh3", "no2", "o3", "pm10", "pm25", "so2"]
EVENT_COLUMN = "event"
NORMAL_LABEL = "none"


# ---------------- LOAD ---------------- #
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [col.lower().strip() for col in df.columns]

# 🔥 Normalize column names to match FEATURES
    rename_map = {
    "pm2.5": "pm25",
    "pm2_5": "pm25",
    "ozone": "o3",
}   

    df.rename(columns=rename_map, inplace=True)

# 🔍 Debug (optional but useful)
    print("Columns after normalization:", df.columns.tolist())
    return df


# ---------------- COMPUTE IMPACT ---------------- #
def compute_impacts(df):

    impacts = {}

    # Baseline (normal condition)
    normal_df = df[df[EVENT_COLUMN] == NORMAL_LABEL]

    if normal_df.empty:
        raise ValueError("No 'none' (baseline) data found")

    normal_mean = normal_df[FEATURES].mean()

    # Events
    events = df[EVENT_COLUMN].unique()

    for event in events:

        if event == NORMAL_LABEL:
            continue

        event_df = df[df[EVENT_COLUMN] == event]

        if event_df.empty:
            continue

        event_mean = event_df[FEATURES].mean()

        impact = (event_mean - normal_mean).to_dict()

        # Convert numpy → float
        impact = {k: float(v) for k, v in impact.items()}

        impacts[event] = impact

    return impacts


# ---------------- SAVE ---------------- #
def save_impacts(impacts):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(impacts, f, indent=4)

    print("✅ Event impacts saved!")
    print(json.dumps(impacts, indent=2))


# ---------------- MAIN ---------------- #
def main():
    print("Loading dataset...")
    df = load_data()

    print("Computing event impacts...")
    impacts = compute_impacts(df)

    print("Saving...")
    save_impacts(impacts)


if __name__ == "__main__":
    main()