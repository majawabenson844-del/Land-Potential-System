import streamlit as st
import pandas as pd
import numpy as np
import joblib

from streamlit_geolocation import streamlit_geolocation

# folium (optional)
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False


# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Benson's Groundwater Potential Mapping System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Load Artifacts
# ===============================
def safe_load(path, name):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load {name} from {path}: {e}")
        return None

model = safe_load("svm_model.pkl", "model")
scaler = safe_load("scaler.pkl", "scaler")
encoder = safe_load("encoder.pkl", "encoder")
selected_features = safe_load("selected_features.pkl", "selected_features")

# ===============================
# Load Dataset (for categories + full feature list)
# ===============================
data = None
try:
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
except Exception as e:
    st.error(f"Could not load dataset augmented_data.csv: {e}")
    data = pd.DataFrame()
    full_features = []

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
    .block-container { padding: 2.5rem; }
    .card {
        background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        border: 1px solid rgba(245, 199, 122, 0.2);
        box-shadow: 0 0 25px rgba(245, 199, 122, 0.15);
    }
    h1, h2, h3 {
        color: #f5c77a !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
    }
    label {
        font-size: 20px !important;
        font-weight: 800 !important;
        color: #f5c77a !important;
    }
    .stSelectbox > div {
        background-color: #121212 !important;
        border: 1px solid #f5c77a !important;
        border-radius: 10px;
        color: white !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #f5c77a, #ffd98e);
        color: black;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-size: 18px;
        font-weight: 800;
        border: none;
        box-shadow: 0 0 15px rgba(245,199,122,0.4);
        transition: 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(245,199,122,0.7);
    }
    .sidebar .sidebar-content { background-color: #0f0f0f; }
    </style>
    """, unsafe_allow_html=True)

apply_gold_black_theme()

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🌍 Groundwater Potential Mapping")
page = st.sidebar.radio("Navigation", ["Home", "Predict", "Model Info", "Feature Guide", "About"])


# ===============================
# HOME
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🌍 Groundwater Potential Mapping Prediction System")
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
# Predict Page (district auto + NO manual centroid coordinates)
# ===============================
elif page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Predict Groundwater Potential")
    st.write("### Select values for the predictors:")

    # --- IMPORTANT ---
    # You said:
    # - dataset has NO district fields and NO district-linked predictors
    # - you do NOT want manually inserted district centroid coordinates
    #
    # Therefore, the only correct way to show "district" is to use reverse geocoding
    # (turn lat/lon into district name). But that requires an external geocoding API
    # and generally needs the "district" name to exist in that API's data.
    #
    # streamlit-geolocation provides lat/lon only; it does NOT give district names.
    #
    # So below we:
    # 1) Detect GPS in real-time
    # 2) Show lat/lon
    # 3) Let you choose district manually is OPTIONAL,
    #    but you asked you want district and NO manual coordinates.
    #
    # Since district-name from GPS is not available without reverse geocoding,
    # we implement reverse geocoding using Nominatim (OpenStreetMap).
    #
    # This avoids any manually inserted centroid coordinates.

    # ---- Reverse geocoding (lat/lon -> district) ----
    # Note: This depends on network access. If it fails, we fall back to "Unknown".
    @st.cache_data(show_spinner=False)
    def reverse_geocode_district(lat, lon):
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="groundwater_app_geocoder")
            loc = geolocator.reverse((lat, lon), zoom=18, language="en")
            if not loc:
                return None
            # Nominatim returns rich address fields; "county"/"state"/"city"/etc vary by place.
            # We'll try common keys.
            addr = loc.raw.get("address", {})
            for key in ["county", "district", "state_district", "state", "region", "municipality"]:
                if key in addr and addr[key]:
                    return addr[key]
            # fallback: sometimes district is part of the display_name
            display_name = loc.raw.get("display_name", "")
            return display_name.split(",")[0].strip() if display_name else None
        except Exception:
            return None

    # Real-time GPS
    if "detected_latlon" not in st.session_state:
        st.session_state["detected_latlon"] = None

    st.markdown("#### Detect your location (real-time GPS)")
    colA, colB = st.columns([2, 1])

    with colA:
        gps = streamlit_geolocation()
        if gps:
            lat = gps.get("latitude")
            lon = gps.get("longitude")
            if lat is not None and lon is not None:
                st.session_state["detected_latlon"] = (lat, lon)
                st.success(f"Real-time: {lat}, {lon}")
    with colB:
        if st.button("↻ Re-detect"):
            st.session_state["detected_latlon"] = None
            st.experimental_rerun()

    latlon = st.session_state.get("detected_latlon", None)

    # Auto-detected district from GPS (no centroids stored in code)
    auto_district = None
    if latlon:
        lat, lon = latlon
        with st.spinner("Finding your district from GPS..."):
            auto_district = reverse_geocode_district(lat, lon)

    # District UI (no manual coordinate insertion; district name comes from geocoding)
    st.markdown("#### District (auto-detected from your GPS)")
    if auto_district:
        st.info(f"✅ District: **{auto_district}**")
    else:
        st.warning("Could not detect district automatically. Please grant location permission, or district name may not be available for your coordinates in the geocoding service.")

    # =========================
    # Feature inputs (predictors)
    # =========================
    st.subheader("🧩 Predictor inputs")

    user_inputs = {}
    if selected_features is None:
        st.error("Selected features are not loaded. Check that 'selected_features.pkl' was loaded successfully.")
    else:
        if data is None or data.empty:
            st.error("Dataset not loaded. Check augmented_data.csv and column setup.")
        else:
            for feature in selected_features:
                if feature in data.columns:
                    options = sorted(data[feature].dropna().unique().tolist())
                    user_inputs[feature] = st.selectbox(f"🔸 {feature}", options, key=f"feat_{feature}")
                else:
                    user_inputs[feature] = st.text_input(f"🔸 {feature} (enter value)", key=f"feat_{feature}")

    # ----- Prediction button -----
    if st.button("✨ Predict Potential"):
        try:
            if model is None or scaler is None or encoder is None or selected_features is None:
                st.error("Required artifacts are not loaded. Check model/scaler/encoder/selected_features files.")
            else:
                full_input = default_values.copy()
                full_input.update(user_inputs)

                # Fill the rest of full_features
                if data is not None and not data.empty:
                    for col in full_features:
                        if col not in full_input:
                            full_input[col] = default_values.get(col, "")

                input_df = pd.DataFrame([full_input])[full_features]

                encoded = encoder.transform(input_df)
                encoded_df = pd.DataFrame(encoded, columns=full_features)

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
# MODEL INFO
# ===============================
elif page == "Model Info":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🧠 Model Information")

    st.write("""
    **Model:** Support Vector Machine (RBF Kernel)  
    **Feature Selection:** Boruta  
    **Scaling:** StandardScaler  
    **Encoding:** OrdinalEncoder  
    **Deployment:** Streamlit Cloud Ready  
    """)

    st.subheader("Selected Predictors:")
    if selected_features:
        for f in selected_features:
            st.write(f"• {f}")
    else:
        st.warning("Selected features list is not available.")
    st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# FEATURE GUIDE
# ===============================
elif page == "Feature Guide":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📘 Feature Guide")

    if data is not None and not data.empty:
        for col in full_features:
            st.subheader(col)
            if col in data.columns:
                try:
                    vals = sorted(data[col].dropna().unique().tolist())
                    st.write("Possible values:")
                    st.write(vals)
                except Exception as e:
                    st.write(f"Could not list values: {e}")
            else:
                st.write(f"Column not found: {col}")
    else:
        st.warning("Dataset not available.")

    st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("ℹ️ About")
    st.write("""
    This system was engineered for:

    • Real-world deployment  
    • Decision support 
    
    Built with reliability and scalability.

             BY BENSON MAJAWA
    """)
    st.markdown("</div>", unsafe_allow_html=True)
