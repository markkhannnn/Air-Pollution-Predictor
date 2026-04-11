import streamlit as st
import pandas as pd
from backend.api_service import fetch_openaq_latest, build_sequence_dataframe
from backend.prediction_service import predict_with_model
from backend.model_manager import list_available_models
from backend.model_utils import load_model_metadata, validate_features, save_model_metadata

st.set_page_config(
    page_title="Air Pollution Forecasting System",
    layout="wide"
)
# ---------------- SIDEBAR NAVIGATION ---------------- #

st.sidebar.title("System Navigation")
page = st.sidebar.radio(
    "Select Mode",
    ["Quick Mode", "Advanced Mode", "Real-Time Mode", "Model Management"]
)

st.title("Air Pollution Prediction System")
st.markdown("---")

# ---------------- QUICK MODE ---------------- #

if page == "Quick Mode":

    st.markdown("## Quick AQI Prediction")
    st.markdown("---")

    st.write("Upload a CSV file containing pollutant values to predict AQI.")

    models = list_available_models()

    if not models:
        st.error("No models found in Models directory.")
        st.stop()

    st.markdown("### Select Model")

    selected_model = st.radio(
    "",
    models,
    horizontal=True
)
    st.markdown("---")
    # Load model metadata
    try:
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(project_root, "Models", selected_model)
        metadata = load_model_metadata(selected_model, base_path=base_path)
        required_features = metadata.get("features", [])
        required_timesteps = metadata.get("timesteps", 24)

        st.info(f"Model expects features: {required_features}")
        st.caption(f"Required timesteps: {required_timesteps}")
    except Exception as e:
       
        st.error(f"Failed to load model metadata: {e}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = df.fillna(method="ffill").fillna(0)
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Set up persistent radio state for Quick Mode
        if "quick_view_mode" not in st.session_state:
            st.session_state.quick_view_mode = "Charts"

        # Store prediction result in session state before visualization
        if st.button("Predict Next Hour AQI", use_container_width=True):
            if len(df) < required_timesteps:
                st.error(f"Dataset must contain at least {required_timesteps} rows of data.")
            else:
                try:
                    # Validate and align features (Quick Mode = non-strict)
                    df_validated, _, _, _ = validate_features(
                        df, required_features, strict=False)
                    st.write(f"Using model: {selected_model}")
                    st.write(f"Input shape: {df_validated.shape}")
                    # Iterative 5-step prediction
                    future_preds = []
                    current_df = df_validated.copy()
                    for _ in range(5):
                        result_step = predict_with_model(selected_model, current_df)
                        future_preds.append(result_step)
                        next_row = pd.DataFrame([result_step["predicted_pollutants"]])
                        current_df = pd.concat([current_df, next_row], ignore_index=True)
                    st.session_state["prediction_result"] = future_preds
                    st.success("Prediction Completed Successfully")
                except Exception as ef:
                    import traceback
                    st.error("Prediction failed")
                    st.error(str(ef))
                    st.text("--- Debug Traceback ---")
                    st.text(traceback.format_exc())

        # Show visualization section if prediction_result exists in session state
        if "prediction_result" in st.session_state:
            results = st.session_state["prediction_result"]
            view_mode = st.radio(
                "Select Visualization",
                ["Charts", "Raw JSON", "AQI Trend"],
                horizontal=True,
                key="quick_view_mode"
            )
            pollutants = results[-1]["predicted_pollutants"]
            pollutant_df = pd.DataFrame({
                "Pollutant": list(pollutants.keys()),
                "Value": list(pollutants.values())
            })
            if view_mode == "Charts":
                st.markdown("### 📊 Pollutant Levels")
                st.bar_chart(pollutant_df.set_index("Pollutant"), use_container_width=True)
                st.caption("X-axis: Pollutants | Y-axis: Concentration")
            elif view_mode == "Raw JSON":
                st.markdown("### 🧾 Raw Output")
                st.json(results)
            elif view_mode == "AQI Trend":
                st.markdown("### 📈 AQI Trend (Simulated)")
                aqi_values = [r["AQI"] for r in results]

                aqi_df = pd.DataFrame({
                    "Hour": [f"+{i+1}" for i in range(len(aqi_values))],
                    "AQI": aqi_values
                })

                st.line_chart(aqi_df.set_index("Hour"))
                st.dataframe(aqi_df)
            st.markdown("### 🌫️ AQI Result")
            latest_result = results[-1]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("AQI", f"{latest_result['AQI']:.2f}")
            with col2:
                st.metric("Category", latest_result["Category"])

            aqi_table = pd.DataFrame({
                "Hour": [f"+{i+1}" for i in range(len(results))],
                "AQI": [r["AQI"] for r in results],
                "Category": [r["Category"] for r in results]
            })

            st.markdown("### 📋 AQI Forecast Table")
            st.dataframe(aqi_table)


# ---------------- ADVANCED MODE ---------------- #

elif page == "Advanced Mode":

    st.markdown("## Advanced Mode")
    

    mode_option = st.radio(
        "Choose Action",
        ["Predict", "Train New Model"],
        horizontal=True
    )
    st.markdown("---")

    # =========================
    # 🔹 PREDICTION MODE
    # =========================
    if mode_option == "Predict":

        #st.markdown("### Advanced Prediction")

        available_models = list_available_models()

        if not available_models:
            st.error("No models found in Models directory.")
            st.stop()

        st.markdown("### Select Model")

        selected_model = st.radio(
    "",
    available_models,
    horizontal=True
)
        st.markdown("---")
        try:
            import os
            project_root = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(project_root, "Models", selected_model)
            metadata = load_model_metadata(selected_model, base_path=base_path)

            required_features = metadata.get("features", [])
            required_timesteps = metadata.get("timesteps", 24)

            st.info(f"Model requires EXACT features: {required_features}")

        except Exception as e:
            st.error(f"Failed to load metadata: {e}")
            st.stop()

        event_type = st.selectbox(
            "Select Event (Optional)",
            ["None", "festival", "traffic", "industrial"]
        )

        if event_type == "None":
            event_type = None
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df = df.fillna(method="ffill").fillna(0)

            st.dataframe(df.head())
            st.write(f"Dataset shape: {df.shape}")

            # Set up persistent radio state for Advanced Mode
            if "adv_view_mode" not in st.session_state:
                st.session_state.adv_view_mode = "Charts"

            # Store prediction result in session state before visualization
            if st.button("Run Prediction", use_container_width=True, key="adv_predict"):
                if len(df) < required_timesteps:
                    st.error(f"Dataset must contain at least {required_timesteps} rows.")
                else:
                    try:
                        ignore_cols = ["timestamp", "date", "datetime", "aqi"]
                        df = df.drop(columns=[col for col in ignore_cols if col in df.columns], errors="ignore")
                        df_validated, missing, extra, is_valid = validate_features(
                            df, required_features, strict=True
                        )
                        if missing:
                            st.error("❌ Missing Required Features")
                            for col in missing:
                                st.write(f"- {col}")
                        if extra:
                            st.warning("⚠️ Extra Columns (ignored)")
                            for col in extra:
                                st.write(f"- {col}")
                        if not is_valid:
                            st.stop()
                        st.success("Validation Passed")
                        # Iterative 5-step prediction for Advanced Mode
                        future_preds = []
                        current_df = df_validated.copy()
                        for _ in range(5):
                            result_step = predict_with_model(
                                selected_model,
                                current_df,
                                event_type
                            )
                            future_preds.append(result_step)
                            next_row = pd.DataFrame([result_step["predicted_pollutants"]])
                            current_df = pd.concat([current_df, next_row], ignore_index=True)
                        st.session_state["prediction_result"] = future_preds
                        st.success("Prediction Completed")
                        if event_type:
                            st.info(f"Event applied: {event_type}")
                    except Exception as ef:
                        import traceback
                        st.error(str(ef))
                        st.text(traceback.format_exc())

            # Show visualization section if prediction_result exists in session state
            if "prediction_result" in st.session_state:
                results = st.session_state["prediction_result"]
                view_mode = st.radio(
                    "Select Visualization",
                    ["Charts", "Raw JSON", "AQI Trend"],
                    horizontal=True,
                    key="adv_view_mode"
                )
                pollutants = results[-1]["predicted_pollutants"]
                pollutant_df = pd.DataFrame({
                    "Pollutant": list(pollutants.keys()),
                    "Value": list(pollutants.values())
                })
                if view_mode == "Charts":
                    st.markdown("### 📊 Pollutant Levels")
                    st.bar_chart(pollutant_df.set_index("Pollutant"), use_container_width=True)
                    st.caption("X-axis: Pollutants | Y-axis: Concentration")
                elif view_mode == "Raw JSON":
                    st.markdown("### 🧾 Raw Output")
                    st.json(results)
                elif view_mode == "AQI Trend":
                    st.markdown("### 📈 AQI Trend (Simulated)")
                    aqi_values = [r["AQI"] for r in results]

                    aqi_df = pd.DataFrame({
                        "Hour": [f"+{i+1}" for i in range(len(aqi_values))],
                        "AQI": aqi_values
                    })

                    st.line_chart(aqi_df.set_index("Hour"))
                    st.dataframe(aqi_df)
                st.markdown("### 🌫️ AQI Result")
                latest_result = results[-1]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AQI", f"{latest_result['AQI']:.2f}")
                with col2:
                    st.metric("Category", latest_result["Category"])

                aqi_table = pd.DataFrame({
                    "Hour": [f"+{i+1}" for i in range(len(results))],
                    "AQI": [r["AQI"] for r in results],
                    "Category": [r["Category"] for r in results]
                })

                st.markdown("### 📋 AQI Forecast Table")
                st.dataframe(aqi_table)

    # =========================
    # 🔹 TRAINING MODE
    # =========================
    elif mode_option == "Train New Model":

        st.markdown("### Train Model")
        

        model_name = st.text_input("Enter Model Name")
        uploaded_file = st.file_uploader("Upload Training Dataset", type=["csv"])

        if uploaded_file and model_name:

            df = pd.read_csv(uploaded_file)

            from backend.model_utils import normalize_dataframe_columns
            df = normalize_dataframe_columns(df)

            # Basic validation
            if len(df) < 50:
                st.error("Dataset too small for training.")
                st.stop()

            if df.shape[1] < 3:
                st.error("Dataset must contain at least 3 features.")
                st.stop()

            st.dataframe(df.head())

            if st.button("Start Training",key="train_model"):

                try:
                    from sklearn.preprocessing import MinMaxScaler
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    from Models.cnn_lstm_model import build_cnn_lstm
                    import numpy as np
                    import joblib
                    import os

                    # Clean
                    # Clean
                    df = df.drop(columns=["timestamp", "aqi"], errors="ignore")
                    df = df.select_dtypes(include=["number"])

                    st.info("Preprocessing dataset...")

                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(df)
                    scaled_df = pd.DataFrame(scaled, columns=df.columns)

                    def create_sequences(data, window=24):
                        X, y = [], []
                        for i in range(len(data) - window):
                            X.append(data.iloc[i:i+window].values)
                            y.append(data.iloc[i+window].values)
                        return np.array(X), np.array(y)

                    X, y = create_sequences(scaled_df)

                    split = int(0.8 * len(X))
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    st.write(f"Training samples: {len(X_train)}")
                    st.write(f"Testing samples: {len(X_test)}")

                    model = build_cnn_lstm(
                        input_shape=(X.shape[1], X.shape[2]),
                        output_dim=y.shape[1]
                    )

                    st.info("Training model...")

                    model.fit(
                        X_train,
                        y_train,
                        epochs=20,
                        batch_size=32,
                        verbose=0
                    )

                    st.success("Training completed!")

                    # Metrics
                    y_pred = model.predict(X_test)

                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    st.markdown("### 📊 Model Performance")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")

                    base_path = os.path.join("Models", model_name)

                    if os.path.exists(base_path):
                        st.warning("Model already exists. Overwriting...")

                    os.makedirs(base_path, exist_ok=True)

                    model.save(os.path.join(base_path, "model.keras"))
                    joblib.dump(scaler, os.path.join(base_path, "scaler.pkl"))

                    save_model_metadata(
                        model_name=model_name,
                        features=list(df.columns),
                        timesteps=24,
                        base_path=base_path
                    )

                    st.success(f"Model '{model_name}' saved successfully!")
                   # st.rerun()

                except Exception as e:
                    st.error(str(e))
# ---------------- REAL-TIME MODE ---------------- #
elif page == "Real-Time Mode":

    st.markdown("## Real-Time AQI Mode")
    st.markdown("---")

    location_id = st.text_input("Enter Location ID")
    api_key_input = st.text_input("Enter OpenAQ API Key", type="password")

    models = list_available_models()
    selected_model = st.selectbox("Select Model", models)

    # Persist view mode selection
    if "realtime_view_mode" not in st.session_state:
        st.session_state.realtime_view_mode = "Graphical"

    # Button triggers fetch, but also allow persistence of result in session state
    if st.button("Fetch & Predict AQI", use_container_width=True) or "realtime_result" in st.session_state:

        try:
            import requests
            import os

            API_KEY = api_key_input if api_key_input else st.secrets.get("OPENAQ_API_KEY", "")
            if not API_KEY:
                st.error("Please provide an API key.")
                st.stop()

            if not location_id:
                st.error("Please enter a location ID.")
                st.stop()

            location_id = location_id.strip()
            headers = {"X-API-Key": API_KEY}

            # ---------------- STEP 1: LOCATION METADATA ---------------- #
            url_loc = f"https://api.openaq.org/v3/locations/{location_id}"
            loc_res = requests.get(url_loc, headers=headers)
            loc_data = loc_res.json()

            if not loc_data.get("results"):
                st.error("Invalid location ID")
                st.stop()

            location_info = loc_data["results"][0]

            # 📍 Station Info
            st.markdown("### 📍 Station Info")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Name", location_info.get("name"))

            with col2:
                st.metric("Locality", location_info.get("locality"))

            with col3:
                st.metric("Country", location_info.get("country", {}).get("name"))

            sensors = location_info.get("sensors", [])

            # Sensor mapping
            sensor_map = {
                s["id"]: s["parameter"]["name"]
                for s in sensors
            }

            # ---------------- STEP 2: LATEST DATA ---------------- #
            url_latest = f"https://api.openaq.org/v3/locations/{location_id}/latest"
            latest_res = requests.get(url_latest, headers=headers)
            latest_data = latest_res.json()

            if not latest_data.get("results"):
                st.error("No recent data available for this station.")
                st.stop()

            latest_entry = latest_data["results"][0]
            latest_time = latest_entry.get("datetime", {})

            # 🕒 Time Info
            st.markdown("### 🕒 Latest Reading")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("UTC Time", latest_time.get("utc"))

            with col2:
                st.metric("Local Time", latest_time.get("local"))

            # ---------------- STEP 3: POLLUTANT MAPPING ---------------- #
            pollutants = {}

            for m in latest_data["results"]:
                sid = m.get("sensorsId")
                val = m.get("value")

                if sid in sensor_map:
                    pollutants[sensor_map[sid]] = val

            # ---------------- VIEW MODE SELECTOR ---------------- #
            view_mode = st.radio(
                "Select View",
                ["Graphical", "Raw JSON"],
                horizontal=True,
                key="realtime_view_mode"
            )

            st.markdown("### Output")

            if view_mode == "Graphical":
                st.markdown("### 📊 Live Pollutant Levels")

                pollutant_df = pd.DataFrame({
                    "Pollutant": list(pollutants.keys()),
                    "Value": list(pollutants.values())
                })

                st.bar_chart(pollutant_df.set_index("Pollutant"), use_container_width=True)
                st.caption("X-axis: Pollutants | Y-axis: Concentration")

            elif view_mode == "Raw JSON":
                st.markdown("### 🧾 Raw Pollutant Data")
                st.json(pollutants)

            # ---------------- STEP 4: MODEL PREP ---------------- #
            df = pd.DataFrame([pollutants])

            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models", selected_model)
            metadata = load_model_metadata(selected_model, base_path=base_path)
            required_features = metadata.get("features", [])

            df = df.reindex(columns=required_features, fill_value=0)
            df_seq = pd.concat([df] * 24, ignore_index=True)

            # ---------------- STEP 5: PREDICT ---------------- #
            future_preds = []
            current_df = df_seq.copy()
            for _ in range(5):
                result_step = predict_with_model(selected_model, current_df)
                future_preds.append(result_step)
                next_row = pd.DataFrame([result_step["predicted_pollutants"]])
                current_df = pd.concat([current_df, next_row], ignore_index=True)
            st.session_state["realtime_result"] = future_preds

            st.success("Prediction Completed")

            # Before displaying, restore result from session state if present
            if "realtime_result" in st.session_state:
                results = st.session_state["realtime_result"]
                latest_result = results[-1]

            # 🌫️ AQI Display
            st.markdown("### 🌫️ AQI Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("AQI", f"{latest_result['AQI']:.2f}")

            with col2:
                st.metric("Category", latest_result["Category"])

            aqi_values = [r["AQI"] for r in results]

            aqi_df = pd.DataFrame({
                "Hour": [f"+{i+1}" for i in range(len(aqi_values))],
                "AQI": aqi_values
            })

            st.markdown("### 📈 Next 5 Hours AQI Forecast")
            st.line_chart(aqi_df.set_index("Hour"))

            aqi_table = pd.DataFrame({
                "Hour": [f"+{i+1}" for i in range(len(results))],
                "AQI": [r["AQI"] for r in results],
                "Category": [r["Category"] for r in results]
            })

            st.markdown("### 📋 AQI Forecast Table")
            st.dataframe(aqi_table)

        except Exception as e:
            st.error(f"Error: {e}")


elif page == "Model Management":

    st.markdown("## Model Management")
    st.markdown("---")

    import os
    import shutil

    models = list_available_models()

    if not models:
        st.warning("No models available.")
        st.stop()

    st.markdown("### Select Model")

    selected_model = st.radio(
    "",
    models,
    horizontal=True
)

    model_path = os.path.join("Models", selected_model)

    st.write(f"Selected Model: **{selected_model}**")

    # ---------------- DELETE ---------------- #

    st.markdown("### Delete Model")

    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False

    if st.button("Delete Model", key="delete_model"):
        st.session_state.confirm_delete = True

    if st.session_state.confirm_delete:
        st.warning("Are you sure you want to delete this model?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Yes, Delete", key="confirm_yes"):
                try:
                  shutil.rmtree(model_path)
                  st.success("Model deleted successfully!")
                  st.session_state.confirm_delete = False
                  st.rerun()
                except Exception as e:
                  st.error(str(e))

        with col2:
         if st.button("Cancel", key="confirm_cancel"):
            st.session_state.confirm_delete = False
            st.info("Deletion cancelled")


    # ---------------- RENAME ---------------- #
    st.markdown("### Rename Model")

    new_name = st.text_input("Enter new model name")

    if st.button("Rename Model", key="rename_model"):

        if not new_name:
            st.error("Enter a valid name")
        else:
            new_path = os.path.join("Models", new_name)

            if os.path.exists(new_path):
                st.error("Model with this name already exists")
            else:
                try:
                    os.rename(model_path, new_path)


                    old_meta = os.path.join(new_path, f"{selected_model}_metadata.json")
                    new_meta = os.path.join(new_path, f"{new_name}_metadata.json")

                    if os.path.exists(old_meta):
                        os.rename(old_meta, new_meta)

                    st.success("Model renamed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
