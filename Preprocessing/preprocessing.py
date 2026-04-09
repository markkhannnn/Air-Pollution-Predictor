import pandas as pd
import os
from aqi_calculator import compute_aqi

def load_all_raw_data(folder_path):
    all_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".csv")
    ]
    
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df


def preprocess_aqi_data(raw_folder_path):
    
    df = load_all_raw_data(raw_folder_path)

    required_indicators = [
        'PM2.5',
        'PM10',
        'NO2',
        'SO2',
        'CO',
        'Ozone',
        'NH3'
    ]

    # Filter required pollutants
    df = df[df['indicator'].isin(required_indicators)]
    # Convert value column to numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    # Create timestamp
    df['timestamp'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        format='%d-%m-%Y %H:%M'
    )

    # Pivot long to wide
    df_wide = df.pivot_table(
        index='timestamp',
        columns='indicator',
        values='value'
    )

    df_wide = df_wide.sort_index()

    # Handle missing values
    df_wide = df_wide.interpolate(method='time')
    df_wide = df_wide.bfill()
    df_wide["AQI"] = df_wide.apply(compute_aqi, axis=1)

    return df_wide


def save_cleaned_data(df_wide, save_path):
    df_wide.to_csv(save_path)