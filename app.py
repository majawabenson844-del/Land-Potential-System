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
# PREDICTION (auto geolocation)
# ===============================
if page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Predict Groundwater Potential")

    st.write("### Select values for the predictors:")

    # Country, province, district selection (kept)
    country = st.selectbox("Select Country", ["Select Country", "Zimbabwe"])
    province = district = None
    if country == "Zimbabwe":
        province = st.selectbox("Select Province", ["Select Province", "Midlands Province", "Masvingo Province"])
        districts = {
            "Midlands Province": ["Chirumhanzu", "Gokwe North", "Gokwe South", "Gweru", "Kwekwe", "Mberengwa", "Shurugwi", "Zvishavane"],
            "Masvingo Province": ["Bikita", "Chiredzi", "Chivi", "Gutu", "Masvingo", "Mwenezi", "Zaka"]
        }
        if province in districts:
            district = st.selectbox("Select District", ["Select District"] + districts[province])

    st.write("The app will attempt to auto-detect your location (browser will ask for permission).")

    # Inject JS that automatically gets geolocation on page load and writes to the hidden input.
    # It retries until the Streamlit hidden input is present in the DOM.
    st.markdown(
        """
        <script>
        async function writeLocationToStreamlit() {
            const findInput = () => {
                // Streamlit gives inputs ids like 'location_input' + suffix; search by starts-with
                return window.parent.document.querySelectorAll('input[id^="auto_location_input"]');
            };
            const tryGetInput = (retries, delay) => new Promise((resolve, reject) => {
                let attempts = 0;
                const timer = setInterval(() => {
                    const el = findInput();
                    if (el && el.length > 0) {
                        clearInterval(timer);
                        resolve(el[0]);
                    } else {
                        attempts++;
                        if (attempts >= retries) {
                            clearInterval(timer);
                            reject("input-not-found");
                        }
                    }
                }, delay);
            });

            try {
                const inputEl = await tryGetInput(50, 200); // retry ~10s max
                if (!navigator.geolocation) {
                    console.warn("Geolocation not supported");
                    return;
                }
                navigator.geolocation.getCurrentPosition((pos) => {
                    inputEl.value = pos.coords.latitude + "," + pos.coords.longitude;
                    inputEl.dispatchEvent(new Event('change', { bubbles: true }));
                }, (err) => {
                    console.warn("Geolocation error:", err);
                }, { enableHighAccuracy: true, timeout: 10000 });
            } catch (e) {
                console.warn("Could not find Streamlit input element for geolocation:", e);
            }
        }
        // Start after a short delay so the Streamlit DOM is ready
        setTimeout(writeLocationToStreamlit, 300);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Hidden text input that JS will populate automatically (stable key)
    location = st.text_input("auto_location", value="", placeholder="Waiting for browser to provide location...", key="auto_location_input")

    # If location is available, parse and show map
    if location:
        try:
            lat_str, lon_str = location.split(",")
            lat, lon = float(lat_str.strip()), float(lon_str.strip())

            m = folium.Map(location=[lat, lon], zoom_start=14)
            folium.Marker(location=[lat, lon], popup="Detected location", icon=folium.Icon(color="blue")).add_to(m)
            folium_static(m)

            st.success(f"Detected location: {lat:.6f}, {lon:.6f}")

        except Exception:
            st.error("Couldn't parse location. Expected format 'lat,lon'.")

    else:
        st.info("Waiting for browser geolocation. Please allow location access when prompted by your browser.")

    # User feature inputs (uses selected_features list you loaded)
    user_inputs = {}
    for feature in selected_features:
        if feature in data.columns:
            options = sorted(data[feature].dropna().unique().tolist())
            user_inputs[feature] = st.selectbox(f"🔸 {feature}", options, index=0)
        else:
            user_inputs[feature] = st.text_input(f"🔸 {feature}", value=default_values.get(feature, ""))

    if st.button("✨ Predict Potential"):
        try:
            # Build input dict using default values and user selections
            full_input = default_values.copy()
            full_input.update(user_inputs)

            # Add numeric lat/lon if available
            if location:
                try:
                    lat_str, lon_str = location.split(",")
                    full_input['Location_Lat'] = float(lat_str.strip())
                    full_input['Location_Lon'] = float(lon_str.strip())
                except Exception:
                    pass

            # Build DataFrame for model
            input_df = pd.DataFrame([full_input])

            # Ensure we pass the correct columns to encoder (selected_features should match)
            # If encoder expects features in a particular order, ensure selected_features matches that.
            encoded = encoder.transform(input_df[selected_features])
            encoded_df = pd.DataFrame(encoded, columns=selected_features)

            selected_df = encoded_df[selected_features]
            scaled = scaler.transform(selected_df)

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
