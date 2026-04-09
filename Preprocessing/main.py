from preprocessing import preprocess_aqi_data, save_cleaned_data

raw_folder = "../Datasets/Raw Datasets"
cleaned_output = "../Datasets/Cleaned Datasets/cleaned_hourly_data.csv"

df_cleaned = preprocess_aqi_data(raw_folder)

save_cleaned_data(df_cleaned, cleaned_output)

print("Preprocessing complete. Cleaned dataset saved.")
