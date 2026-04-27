import streamlit as st
import pandas as pd
import joblib
from geopy.geocoders import Nominatim

# -----------------------------------------------------------------------
# GPS bridge — history of fixes
#
# ORIGINAL BUG: The `streamlit-geolocation` package always returned None
# for latitude / longitude, so the map never received a position.
#
# FIRST ATTEMPT (failed): rewrote the button using `st.components.v1.html()`
# and round-tripped the coordinates through the URL with st.query_params.
# Setting `window.parent.location.href` triggered a full page navigation,
# which Streamlit treats as a fresh load — the sidebar reset to "Home"
# and the Predict-page code never re-ran.
#
# SECOND ATTEMPT (failed): used postMessage + history.pushState +
# popstate to update the URL silently. Streamlit does not consistently
# react to programmatically dispatched popstate events; the rerun
# fired sometimes, not always.
#
# THIRD ATTEMPT (failed): wrote the coordinates from JS into a hidden
# `st.text_input` using React's nativeInputValueSetter and a synthetic
# 'input' event. This is fragile because Streamlit binds inputs through
# its own state machine and does not reliably observe synthetic events,
# AND because querySelectorAll('input[type="text"]') matches every text
# input on the page (including predictor inputs), so the value
# frequently landed in the wrong field.
#
# CURRENT FIX: use the `streamlit-js-eval` package's get_geolocation().
# It is a properly declared Streamlit custom component; its iframe is
# configured with the geolocation permission policy and the value comes
# back through Streamlit.setComponentValue() — the official supported
# channel for component-to-Python data flow. No DOM hacking, no
# synthetic events, no URL round-trip.
# -----------------------------------------------------------------------
import streamlit.components.v1 as components  # kept (unused) — harmless
from streamlit_js_eval import get_geolocation  # NEW: official component bridge

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


def getLocDetails(lat,long):
# Initialize Nominatim API
    geolocator = Nominatim(user_agent="geoapi_exercise")
    
    # Coordinates (Latitude, Longitude)
    # coords = "25.594095, 85.137566"
    # coords = "-19.447881,29.813125"
    coords = f"{lat},{long}"
    # Get location information
    location = geolocator.reverse(coords)
    
    # Access the address dictionary
    address = location.raw['address']
    
    # Safely extract country
    country = address.get('country', '')
    state = address.get('state', '')
    city = address.get('city', '')
    
    return f"Country: {country}, Province: {state}, City: {city}"


# print(address)


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

    st.subheader("📍 Detect your location (real-time GPS)")

    # -------------------------------------------------------------------
    # GPS bridge — final implementation
    #
    # WHAT THIS DOES, IN PLAIN ENGLISH:
    #
    # The browser's Geolocation API is JavaScript-only — Python on the
    # server cannot call it directly.  To bridge the two we use the
    # `streamlit-js-eval` package's `get_geolocation()` helper.  It
    # registers a real Streamlit custom component whose iframe has the
    # geolocation permission policy set; when the JS inside it gets a
    # position fix it returns it to Python through the official
    # Streamlit.setComponentValue() channel.  No DOM hacking, no event
    # simulation, no URL round-trip — none of the failure modes the
    # earlier attempts ran into.
    #
    # FLOW:
    #   1. User clicks "Detect My Location"  →  we set geo_active=True
    #      and bump geo_request_id (the component caches by component_key,
    #      so changing the key on each request is what re-triggers the
    #      browser prompt).
    #   2. Script reruns.  get_geolocation(...) renders the component
    #      and returns None (browser is still prompting).
    #   3. User accepts → component sends {coords: {...}} back to Python.
    #      Streamlit reruns the script automatically.
    #   4. get_geolocation(...) now returns the dict; we extract lat/lon,
    #      stash in session_state, print() them to the console, and clear
    #      geo_active so the component stops asking on idle reruns.
    # -------------------------------------------------------------------

    # Per-request id so streamlit-js-eval re-prompts the browser each
    # time the user clicks Detect or Re-detect (the helper caches its
    # result against component_key).
    if "geo_request_id" not in st.session_state:
        st.session_state["geo_request_id"] = 0

    # Are we currently waiting for / processing a geolocation fix?
    if "geo_active" not in st.session_state:
        st.session_state["geo_active"] = False

    geo_col1, geo_col2 = st.columns([2, 1])

    with geo_col1:
        if st.button("📍 Detect My Location"):
            st.session_state["geo_request_id"] += 1
            st.session_state["geo_active"] = True
            st.session_state["detected_latlon"] = None

    with geo_col2:
        if st.button("↻ Re-detect"):
            st.session_state["geo_request_id"] += 1
            st.session_state["geo_active"] = True
            st.session_state["detected_latlon"] = None

    # Only render the geolocation component while a request is active.
    # On its first render this returns None (browser still prompting);
    # once the user accepts, the component triggers a rerun and returns
    # a dict like {"coords": {"latitude": ..., "longitude": ...}, ...}.
    if st.session_state["geo_active"]:
        loc = get_geolocation(
            component_key=f"geo_request_{st.session_state['geo_request_id']}"
        )

        if loc and isinstance(loc, dict) and loc.get("coords"):
            detected_lat = loc["coords"]["latitude"]
            detected_lon = loc["coords"]["longitude"]

            st.session_state["detected_latlon"] = (detected_lat, detected_lon)
            st.session_state["geo_active"] = False

            # ✅ Coordinates are now in Python — print to the streamlit
            # server console as the user explicitly requested.
            # flush=True so it appears immediately rather than buffered.
            print(
                f"[GPS] Button clicked — Latitude: {detected_lat}, "
                f"Longitude: {detected_lon}",
                flush=True,
            )

    # Status line shown to the user in the app itself.
    latlon = st.session_state.get("detected_latlon", None)
    if latlon:
        val = getLocDetails(latlon[0], latlon[1])
        st.success(f"📌 Location detected: Longitude: {latlon[0]:.6f}, Latitude: {latlon[1]:.6f}  " + val)
        # st.success(val)
    elif st.session_state["geo_active"]:
        st.info("⏳ Waiting for the browser… please allow location access when it prompts you.")
    else:
        st.info("Click 📍 Detect My Location, then allow location access when the browser prompts.")

    # latlon is already set above; the Folium map block below reads it.

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
