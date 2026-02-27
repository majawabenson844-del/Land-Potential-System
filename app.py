import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Benson's GWP  Mapping Estimator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Load Artifacts
# ===============================
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
selected_features = joblib.load("selected_features.pkl")

# ===============================
# Load Dataset (for categories + full feature list)
# =================================
data = pd.read_csv("augmented_data.csv")
data.columns = [
    'Decision',
    'Soil.Texture',
    'Soil.Colour',
    'Geological.Features',
    'Elevation',
    'Natural.vegitation..tree..vigour',
    'Natural.vegitation..tree..height',
    'Drainage.Density'
]

full_features = [col for col in data.columns if col != "Decision"]

# ===============================
# Default values for hidden features
# ===============================
default_values = {col: data[col].mode()[0] for col in full_features}

# ===============================
# GOLD + BLACK PREMIUM THEME
# ===============================
def apply_gold_black_theme():
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #0b0b0b;
        color: #f5c77a;
    }
    /* Add your existing CSS styles here */
    </style>
    """, unsafe_allow_html=True)

apply_gold_black_theme()

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🌍 GWP  Mapping Estimator")
page = st.sidebar.radio("Navigation", ["Home", "Predict", "Model Info", "Feature Guide", "About"])

# ===============================
# HOME
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🌍 GWP  Mapping Estimator")
    st.write("""
    A **decision support system** built with:

    • Boruta Feature Selection  
    • Ordinal Encoding  
    • Standard Scaling  
    • Support Vector Machines 

    Designed for **real-world deployment**.

              By BENSON T MAJAWA
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# PREDICTION
# ===============================
elif page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Predict Groundwater Potential")

    st.write("### Select values for the predictors:")

    # Country selection
    country = st.selectbox("Select Country", ["Select Country", "Zimbabwe"])

    if country == "Zimbabwe":
        province = st.selectbox("Select Province", ["Select Province", "Midlands Province", "Masvingo Province"])
        districts = {
            "Midlands Province": ["Chirumhanzu", "Gokwe North", "Gokwe South", "Gweru", "Kwekwe", "Mberengwa", "Shurugwi", "Zvishavane"],
            "Masvingo Province": ["Bikita", "Chiredzi", "Chivi", "Gutu", "Masvingo", "Mwenezi", "Zaka"]
        }
        if province in districts:
            district = st.selectbox("Select District", ["Select District"] + districts[province])
        else:
            district = None  # If no valid province is selected

    # Function to get user location without an API
    def get_location():
        st.markdown("""
        <script>
        async function getLocation() {
            // Check if geolocation is available and supported
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    const locationInfo = lat + ", " + lon;
                    const hiddenInput = document.getElementById('location');
                    hiddenInput.value = locationInfo; // Set the input value
                }, function() {
                    alert("Unable to retrieve your location.");
                });
            } else {
                alert("Geolocation is not supported by this browser.");
            }
        }
        getLocation();
        </script>
        <input type="hidden" id="location" value="">
        """, unsafe_allow_html=True)

    get_location()

    # Input for user's location
    location = st.text_input("Your Location (Lat, Lon)", value="", placeholder="Automatically fetched location", key='location_input', disabled=True)

    user_inputs = {}

    for feature in selected_features:
        options = sorted(data[feature].dropna().unique().tolist())
        user_inputs[feature] = st.selectbox(f"🔸 {feature}", options)

    if st.button("✨ Predict Potential"):
        try:
            full_input = default_values.copy()
            full_input.update(user_inputs)

            # Add location if available
            if location and location != "":
                full_input['Location'] = location  # Ensure 'Location' is part of your model's features

            input_df = pd.DataFrame([full_input])[full_features]

            # Encode
            encoded = encoder.transform(input_df)
            encoded_df = pd.DataFrame(encoded, columns=full_features)

            # Select Boruta features
            selected_df = encoded_df[selected_features]

            # Scale
            scaled = scaler.transform(selected_df)

            # Predict
            pred = model.predict(scaled)[0]
            probs = model.predict_proba(scaled)[0]

            st.markdown("---")

            if pred == 1:
                st.success("🌱 High Potential Area")
            else:
                st.error("⚠️ Low Potential Area")

            col1, col2 = st.columns(2)
            col1.metric("High Potential Confidence", f"{probs[1]*100:.2f}%")
            col2.metric("Low Potential Confidence", f"{probs[0]*100:.2f}%")

        except Exception as e:
            st.error(f"System Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# MODEL INFO, FEATURE GUIDE, ABOUT
# ===============================
# The existing Model Info, Feature Guide, and About sections remain unchanged.
# Add the code for those sections below.

# (Include the other existing sections here)
