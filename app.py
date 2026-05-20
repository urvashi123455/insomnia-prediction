import streamlit as st
import joblib
import numpy as np
from scipy.signal import welch
import mne
import tempfile
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sleep Stage Detection",
    layout="centered"
)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("sleep_model.pkl")
scaler = joblib.load("scaler.pkl")

# ============================================================
# LABELS
# ============================================================

sleep_labels = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "Deep",
    4: "REM"
}

# ============================================================
# UI
# ============================================================

st.title("Sleep Stage Detection System")

st.write("Upload EDF EEG file")

uploaded_file = st.file_uploader(
    "Choose EDF File",
    type=["edf"]
)

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    st.info("Processing EEG file...")

    temp_path = None

    try:

        # Save temp EDF
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".edf"
        ) as tmp:

            tmp.write(uploaded_file.read())

            temp_path = tmp.name

        # Load EDF
        raw = mne.io.read_raw_edf(
            temp_path,
            preload=True,
            verbose=False
        )

        # Small EEG segment
        signal = raw.get_data(
            start=0,
            stop=3000
        )[0]

        # Feature extraction
        freqs, psd = welch(
            signal,
            fs=raw.info['sfreq']
        )

        delta = np.mean(
            psd[(freqs >= 0.5) & (freqs < 4)]
        )

        theta = np.mean(
            psd[(freqs >= 4) & (freqs < 8)]
        )

        alpha = np.mean(
            psd[(freqs >= 8) & (freqs < 13)]
        )

        beta = np.mean(
            psd[(freqs >= 13) & (freqs < 30)]
        )

        features = [[
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            delta,
            theta,
            alpha,
            beta
        ]]

        # Scale
        X_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(X_scaled)[0]

        result = sleep_labels.get(
            int(prediction),
            "Unknown"
        )

        # Display Result
        st.success(
            f"Predicted Sleep Stage: {result}"
        )

    except Exception as e:

        st.error(str(e))

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)