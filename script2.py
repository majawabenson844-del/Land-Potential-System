import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and encoder
model = joblib.load("svm_model_boruta.pkl")
scaler = joblib.load("scaler_boruta.pkl")
encoder = joblib.load("ordinal_encoder.pkl")

# Boruta-selected features (variable-level)
important_features = [
    'Geological.Features',
    'Elevation',
    'Natural.vegitation..tree..vigour',
    'Natural.vegitation..tree..height',
    'Drainage.Density'
]

# Full original feature list
original_features = [
    'Soil.Texture',
    'Soil.Colour',
    'Geological.Features',
    'Elevation',
    'Natural.vegitation..tree..vigour',
    'Natural.vegitation..tree..height',
    'Drainage.Density'
]

st.title("🌍Groundwater  Potential Mapping (SVM + Boruta)")
st.markdown("Predict **High vs Low Potential** using Boruta-selected features and an SVM model trained with ordinal encoding.")

# --- Upload CSV for batch predictions ---
uploaded_file = st.file_uploader("Upload a CSV file with predictors", type=["csv"])
if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", df_new.head())

    # Apply encoder
    df_encoded = pd.DataFrame(
        encoder.transform(df_new),
        columns=original_features
    )

    # Select Boruta features
    df_selected = df_encoded[important_features]

    # Scale
    df_scaled = scaler.transform(df_selected)

    # Predict
    preds = model.predict(df_scaled)
    df_new["Prediction"] = np.where(preds == 1, "High Potential", "Low Potential")

    st.write("Predictions:", df_new)

# --- Manual input for single prediction ---
st.header("🔍 Try a single prediction")
inputs = {}
for feat in original_features:
    inputs[feat] = st.text_input(f"{feat}")

if st.button("Predict"):
    X_input = pd.DataFrame([inputs])

    # Apply encoder
    X_encoded = pd.DataFrame(
        encoder.transform(X_input),
        columns=original_features
    )

    # Select Boruta features
    X_selected = X_encoded[important_features]

    # Scale and predict
    X_scaled = scaler.transform(X_selected)
    pred = model.predict(X_scaled)[0]
    st.success(f"Prediction: {'High Potential' if pred == 1 else 'Low Potential'}")
