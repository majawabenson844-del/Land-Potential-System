import streamlit as st
import pandas as pd
import joblib

from streamlit_geolocation import streamlit_geolocation

# Optional map preview
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
    initial_sidebar_state="expanded",
)

# ===============================
# Theme
# ===============================
st.markdown(
    """
<style>
body, .stApp { background-color:#0b0b0b; color:#f5c77a; }
.block-container { padding: 2.5rem; }
.card {
  background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
  border-radius: 18px;
  padding: 25px;
  margin-bottom: 20px;
  border: 1px solid rgba(245, 199, 122, 0.2);
  box-shadow: 0 0 25px rgba(245, 199, 122, 0.15);
}
h1, h2, h3 { color:#f5c77a !important; font-weight:800 !important; letter-spacing:1px; }
label { font-size:20px !important; font-weight:800 !important; color:#f5c77a !important; }
.stSelectbox > div { background-color:#121212 !important; border:1px solid #f5c77a !important; border-radius:10px; color:white !important; }
.stButton button {
  background: linear-gradient(90deg, #f5c77a, #ffd98e);
  color:black; border-radius:12px;
  padding:0.8rem 1.5rem;
  font-size:18px; font-weight:800;
  border:none; box-shadow: 0 0 15px rgba(245,199,122,0.4);
  transition:0.3s;
}
.stButton button:hover { transform:scale(1.05); box-shadow: 0 0 25px rgba(245,199,122,0.7); }
.sidebar .sidebar-content { background-color:#0f0f0f; }
</style>
""",
    unsafe_allow_html=True,
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
# Load Dataset (for feature options)
# ===============================
data = None
full_features = []
default_values = {}

try:
    data = pd.read_csv("augmented_data.csv")

    # If your CSV has a different order/names, edit these lines:
    data.columns = [
        "Decision",
        "Soil.Texture",
        "Soil.Colour",
        "Geological.Features",
        "Elevation",
        "Natural.vegitation..tree..vigour",
        "Natural.vegitation..tree..height",
        "Drainage.Density",
    ]

    full_features = [c for c in data.columns if c != "Decision"]

    for col in full_features:
        try:
            default_values[col] = data[col].mode(dropna=True).iloc[0]
        except Exception:
            default_values[col] = ""
except Exception as e:
    st.error(f"Could not load augmented_data.csv: {e}")
    data = pd.DataFrame()
    full_features = []
    default_values = {}

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🌍 Groundwater Potential Mapping")
page = st.sidebar.radio(
    "Navigation", ["Home", "Predict", "Model Info", "Feature Guide", "About"]
)


# ===============================
# HOME
# ===============================
if page == "Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🌍 Groundwater Potential Mapping Prediction System")
    st.write(
        """
A **decision support system** built with:

• Boruta Feature Selection  
• Ordinal Encoding  
• Standard Scaling  
• Support Vector Machines  

Designed for **real-world deployment**.

By BENSON T MAJAWA
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# PREDICT
# ===============================
elif page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Predict Groundwater Potential")
    st.write("### Select values for the predictors:")

    # session state (safe approach; no experimental_rerun, no key= in geolocation)
    if "detected_latlon" not in st.session_state:
        st.session_state["detected_latlon"] = None

    if "refresh" not in st.session_state:
        st.session_state["refresh"] = 0

    st.subheader("📍 Detect your location (real-time GPS)")

    geo_col1, geo_col2 = st.columns([2, 1])

    with geo_col1:
        # IMPORTANT: your installed streamlit-geolocation does NOT accept key=
        gps = streamlit_geolocation()

        if gps:
            lat = gps.get("latitude")
            lon = gps.get("longitude")
            if lat is not None and lon is not None:
                st.session_state["detected_latlon"] = (lat, lon)
                st.success(f"Real-time: {lat}, {lon}")

        # Debug (helps confirm GPS works)
        st.write("GPS raw (latitude/longitude):", gps)

    with geo_col2:
        if st.button("↻ Re-detect"):
            st.session_state["detected_latlon"] = None
            st.session_state["refresh"] += 1
            st.rerun()

    latlon = st.session_state.get("detected_latlon", None)

    # ---- Map Preview with a clearly visible marker ----
    if FOLIUM_AVAILABLE:
        # Zimbabwe-ish fallback if GPS missing
        base_location = latlon if latlon else (-19.0, 29.0)

        m = folium.Map(location=base_location, zoom_start=14, tiles="OpenStreetMap")

        if latlon:
            lat, lon = latlon

            # Big circle marker (high visibility)
            folium.CircleMarker(
                location=[lat, lon],
                radius=16,
                color="#00ffea",
                fill=True,
                fill_color="#00ffea",
                fill_opacity=0.95,
                popup="You are here",
            ).add_to(m)

            # Text label marker using DivIcon
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-size:14px;
                        font-weight:900;
                        color:#000;
                        background:#00ffea;
                        padding:6px 10px;
                        border-radius:10px;
                        box-shadow: 0 0 14px rgba(0,255,234,0.6);
                        border:2px solid #ffffff;
                    ">
                    YOU
                    </div>
                    """
                ),
            ).add_to(m)

        st_folium(m, width=700, height=380)

    else:
        st.info("Map preview disabled (install folium + streamlit-folium if you want).")

    st.caption("If you don’t see the marker, it means GPS returned nothing (check the GPS raw line above).")

    st.markdown("---")
    st.subheader("🧩 Predictor inputs")

    if selected_features is None:
        st.error("Selected features are not loaded. Check selected_features.pkl.")
    elif data is None or data.empty:
        st.error("Dataset not loaded correctly. Check augmented_data.csv.")
    else:
        user_inputs = {}

        for feature in selected_features:
            # If feature exists in your dataset, use select options
            if feature in data.columns:
                options = sorted(data[feature].dropna().unique().tolist())
                user_inputs[feature] = st.selectbox(f"🔸 {feature}", options, key=f"feat_{feature}")
            else:
                # Otherwise allow manual entry (string)
                user_inputs[feature] = st.text_input(
                    f"🔸 {feature} (enter value)",
                    key=f"feat_{feature}",
                    value=str(default_values.get(feature, "")),
                )

        if st.button("✨ Predict Potential"):
            try:
                if model is None or scaler is None or encoder is None:
                    st.error("Model/scaler/encoder not loaded. Check pickle files.")
                else:
                    # Build full input for ALL features used by the pipeline
                    full_input = dict(default_values)
                    full_input.update(user_inputs)

                    # Ensure all full_features exist
                    for col in full_features:
                        if col not in full_input:
                            full_input[col] = default_values.get(col, "")

                    # Order columns exactly
                    input_df = pd.DataFrame([full_input])[full_features]

                    # Encode + scale + select
                    encoded = encoder.transform(input_df)
                    encoded_df = pd.DataFrame(encoded, columns=full_features)

                    selected_df = encoded_df[selected_features]
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
# MODEL INFO
# ===============================
elif page == "Model Info":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🧠 Model Information")
    st.write(
        """
**Model:** Support Vector Machine (SVM)  
**Feature Selection:** Boruta  
**Scaling:** StandardScaler  
**Encoding:** OrdinalEncoder
"""
    )

    st.subheader("Selected Predictors:")
    if selected_features:
        for f in selected_features:
            st.write(f"• {f}")
    else:
        st.warning("Selected feature list is not available. Check selected_features.pkl.")

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
                st.write(f"Column '{col}' not found in loaded dataset.")
    else:
        st.warning("Dataset not available. Check augmented_data.csv.")

    st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("ℹ️ About")
    st.write(
        """
This system was engineered for:

• Real-world deployment  
• Decision support  

Built with reliability and scalability.

BY BENSON MAJAWA
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)
