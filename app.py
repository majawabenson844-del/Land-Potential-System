import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import folium_static

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Benson's GWP Mapping Estimator",
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
    </style>
    """, unsafe_allow_html=True)

apply_gold_black_theme()

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🌍 GWP Mapping Estimator")
page = st.sidebar.radio("Navigation", ["Home", "Predict", "Model Info", "Feature Guide", "About"])

# ===============================
# HOME
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🌍 GWP Mapping Estimator")
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

    # ===============================
    # Function to get user location
    # ===============================
    def get_location():
        st.markdown("""
        <script>
            async function getLocation() {
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(function(position) {
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;
                        const locationInfo = lat + "," + lon;
                        const hiddenInput = document.getElementById('location_input');
                        hiddenInput.value = locationInfo; 
                        // Trigger Streamlit re-run to show map
                        const mapButton = document.getElementById('map-button');
                        if (mapButton) mapButton.click();
                    }, function() {
                        alert("Unable to retrieve your location. Please allow location access in your browser.");
                    });
                } else {
                    alert("Geolocation is not supported by this browser.");
                }
            }
            getLocation();
        </script>
        <input type="hidden" id="location_input" value="">
        """, unsafe_allow_html=True)

    # Get user location
    get_location()

    # Input for user's location
    location = st.text_input("Your Location (Lat, Lon)", value="", placeholder="Automatically fetched location", key='location_input', disabled=True)

    # User feature inputs
    user_inputs = {}
    for feature in selected_features:
        options = sorted(data[feature].dropna().unique().tolist())
        user_inputs[feature] = st.selectbox(f"🔸 {feature}", options)

    # Button to show the map based on location
    if st.button("Show My Location", key="map-button"):
        if location and location != "":
            lat, lon = map(float, location.split(","))

            # Create a Folium map centered on user location
            m = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker(location=[lat, lon], popup="You are here!", icon=folium.Icon(color='blue')).add_to(m)

            # Render the map in Streamlit
            folium_static(m)
        else:
            st.error("Location not available. Please allow location access in your browser.")

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
elif page == "Model Info":
    # Your existing Model Info code goes here
    pass

elif page == "Feature Guide":
    # Your existing Feature Guide code goes here
    pass

elif page == "About":
    # Your existing About code goes here
    pass
