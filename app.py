import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import re
import json

# Try to import folium and streamlit_folium
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

# Example: optional district centroids (latitude, longitude).
# You can expand this with real centroid coordinates.
DISTRICT_CENTROIDS = {
    "Chirumhanzu": (-19.0, 31.9),
    "Gokwe North": (-18.3, 27.0),
    "Gokwe South": (-18.6, 27.1),
    "Gweru": (-19.45, 29.82),
    "Kwekwe": (-18.92, 29.81),
    "Mberengwa": (-20.2, 30.2),
    "Shurugwi": (-19.6, 30.0),
    "Zvishavane": (-20.33, 30.03),
    "Bikita": (-20.96, 31.58),
    "Chiredzi": (-21.05, 31.67),
    "Chivi": (-20.2, 31.4),
    "Gutu": (-20.46, 30.99),
    "Masvingo": (-20.07, 30.83),
    "Mwenezi": (-21.55, 31.6),
    "Zaka": (-20.33, 31.6),
}

# ===============================
# Default values for hidden features
# ===============================
default_values = {}
if not data.empty:
    for col in full_features:
        try:
            default_values[col] = data[col].mode()[0]
        except Exception:
            default_values[col] = ""

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

    .block-container {
        padding: 2.5rem;
    }

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

    .sidebar .sidebar-content {
        background-color: #0f0f0f;
    }

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

# ----------------- Predict page UI (folium integrated) -----------------
elif page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Predict Groundwater Potential")
    st.write("### Select values for the predictors:")

    # Country / Province / District UI
    country = st.selectbox("Select Country", ["Select Country", "Zimbabwe"])
    province = None
    district = None
    if country == "Zimbabwe":
        province = st.selectbox("Select Province", ["Select Province", "Midlands Province", "Masvingo Province"])
        districts = {
            "Midlands Province": ["Chirumhanzu", "Gokwe North", "Gokwe South", "Gweru", "Kwekwe", "Mberengwa", "Shurugwi", "Zvishavane"],
            "Masvingo Province": ["Bikita", "Chiredzi", "Chivi", "Gutu", "Masvingo", "Mwenezi", "Zaka"]
        }
        if province in districts:
            district = st.selectbox("Select District", ["Select District"] + districts[province])

    # Geolocation UI
    st.markdown("#### Detect your location (optional)")
    geo_col1, geo_col2 = st.columns([2,1])

    detected = st.session_state.get("detected_location", "")

    with geo_col1:
        loc_display = st.text_input("Your Location (lat, lon)", value=detected, placeholder="Click 'Get Location' to auto-detect", key="location_input")

    with geo_col2:
        if st.button("📍 Get Location"):
            get_geo_html = """
            <html>
              <body>
                <script>
                  function send(msg) {
                    const pre = document.createElement('pre');
                    pre.id = 'out';
                    pre.innerText = JSON.stringify(msg);
                    document.body.appendChild(pre);
                  }

                  function fail(message) {
                    send({error: true, message: message});
                  }

                  if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                      function(pos) {
                        send({lat: pos.coords.latitude, lon: pos.coords.longitude});
                      },
                      function(err) {
                        fail(err.message || 'Geolocation error');
                      },
                      {enableHighAccuracy: true, timeout: 10000}
                    );
                  } else {
                    fail('Geolocation not supported');
                  }
                </script>
              </body>
            </html>
            """
            try:
                raw = components.html(get_geo_html, height=200)
                m = re.search(r"<pre id=\"out\">(.*?)</pre>", raw, re.S)
                if m:
                    payload = json.loads(m.group(1))
                    if isinstance(payload, dict) and not payload.get("error", False):
                        coords = f"{payload['lat']}, {payload['lon']}"
                        st.session_state["detected_location"] = coords
                        st.experimental_rerun()
                    else:
                        err_msg = payload.get("message", "Unknown geolocation error")
                        st.warning(f"Geolocation failed: {err_msg}")
                else:
                    st.warning("Could not read geolocation response. Please allow location permission or paste coordinates manually.")
            except Exception as e:
                st.error(f"Geolocation component error: {e}")

    # Manual fallback
    st.markdown("If automatic detection fails, paste coordinates in the format: lat, lon (e.g., -19.0154, 29.1549).")
    manual_location = st.text_input("Or paste coordinates here", value="", placeholder="-19.0154, 29.1549", key="manual_loc")
    location = st.session_state.get("detected_location", "") or (manual_location.strip() if manual_location else "")

    # Show folium map if available
    if not FOLIUM_AVAILABLE:
        st.warning("folium or streamlit_folium not installed. To enable map display, install via: pip install folium streamlit-folium")
    else:
        # Decide map center and markers
        map_center = (-19.0, 29.0)  # default center for Zimbabwe-ish
        m = folium.Map(location=map_center, zoom_start=6, tiles="OpenStreetMap")

        # If user provided manual or detected location, parse and add marker
        def parse_coords(s):
            try:
                parts = [p.strip() for p in s.split(",")]
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    return lat, lon
            except Exception:
                return None

        user_point = None
        if location:
            parsed = parse_coords(location)
            if parsed:
                user_point = parsed
                folium.Marker(location=parsed, tooltip="Your Location", icon=folium.Icon(color="blue", icon="user")).add_to(m)
                m.location = parsed
                m.zoom_start = 12

        # If district selected and centroid known, add marker and (if no user point) center map there
        if district and district != "Select District":
            centroid = DISTRICT_CENTROIDS.get(district)
            if centroid:
                folium.Marker(location=centroid, tooltip=f"District: {district}", icon=folium.Icon(color="green", icon="map-marker")).add_to(m)
                if not user_point:
                    m.location = centroid
                    m.zoom_start = 10

        # Render map
        st.caption("Map: detected or manual location (blue) and selected district centroid (green)")
        st_folium(m, width=700, height=400)

    # ---------------- Feature inputs ----------------
    user_inputs = {}
    if selected_features is None:
        st.error("Selected features are not loaded. Check that 'selected_features.pkl' was loaded successfully.")
    else:
        for feature in selected_features:
            if feature in data.columns:
                options = sorted(data[feature].dropna().unique().tolist())
                user_inputs[feature] = st.selectbox(f"🔸 {feature}", options, key=f"feat_{feature}")
            else:
                user_inputs[feature] = st.text_input(f"🔸 {feature} (enter value)", value=default_values.get(feature, ""), key=f"feat_{feature}")

    # ----- Prediction button -----
    if st.button("✨ Predict Potential"):
        try:
            missing = []
            for name in ("encoder", "scaler", "model"):
                if globals().get(name, None) is None:
                    missing.append(name)
            if missing:
                st.error(f"Required objects not loaded: {', '.join(missing)}")
            else:
                full_input = default_values.copy()
                full_input.update(user_inputs)

                if location and 'Location' in full_features:
                    full_input['Location'] = location

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
    try:
        if selected_features:
            for f in selected_features:
                st.write(f"• {f}")
        else:
            st.warning("Selected feature list is not available. Ensure 'selected_features.pkl' was loaded successfully.")
    except Exception as e:
        st.error(f"Error while displaying selected predictors: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# FEATURE GUIDE
# ===============================
elif page == "Feature Guide":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📘 Feature Guide")

    try:
        if full_features and isinstance(data, pd.DataFrame):
            for col in full_features:
                st.subheader(col)
                st.write("Possible values:")
                if col in data.columns:
                    try:
                        vals = sorted(data[col].dropna().unique().tolist())
                        st.write(vals)
                    except Exception as e:
                        st.write(f"Could not list values for {col}: {e}")
                else:
                    st.write(f"Column '{col}' not found in loaded dataset.")
        else:
            st.warning("Feature list or dataset not available. Ensure 'augmented_data.csv' loaded and columns were set correctly.")
    except Exception as e:
        st.error(f"Error while building Feature Guide: {e}")

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
