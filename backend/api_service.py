import requests
import pandas as pd

API_KEY = "b7f3db3e4dc135c585bea4e7f6cee4ba2252a688e99f17800ef85c9547258383"


POLLUTANT_MAPPING = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "o3": "Ozone",
    "nh3": "NH3"
}


def fetch_openaq_latest(city):

    headers = {
        "X-API-Key": API_KEY,
        "Accept": "application/json",
        "User-Agent": "curl/7.68.0"
    }

    # STEP 1 — Get location ID using correct filter
    loc_url = "https://api.openaq.org/v3/locations"

    loc_params = {
        
        "limit": 1
    }

    loc_response = requests.get(loc_url, headers=headers, params=loc_params)

    if loc_response.status_code != 200:
        raise ValueError(f"Location fetch failed: {loc_response.text}")

    loc_data = loc_response.json()

    if not loc_data.get("results"):
        raise ValueError("City not found.")

    location_id = loc_data["results"][0]["id"]

    # STEP 2 — Fetch measurements for that location
    meas_url = "https://api.openaq.org/v3/measurements"

    meas_params = {
        "location_id": location_id,
        "limit": 50,
        "sort": "desc",
        "order_by": "datetime"
    }

    meas_response = requests.get(meas_url, headers=headers, params=meas_params)

    if meas_response.status_code != 200:
        raise ValueError(f"Measurement fetch failed: {meas_response.text}")

    meas_data = meas_response.json()

    pollutant_data = {}

    for item in meas_data.get("results", []):
        param = item.get("parameter", {}).get("name")
        value = item.get("value")

        if param in POLLUTANT_MAPPING:
            mapped_name = POLLUTANT_MAPPING[param]
            pollutant_data[mapped_name] = value

    if not pollutant_data:
        raise ValueError("No required pollutants found in API response.")

    return pollutant_data


def build_sequence_dataframe(pollutant_data):
    return pd.DataFrame([pollutant_data] * 24)