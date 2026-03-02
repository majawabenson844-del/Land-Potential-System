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
# PREDICT (fast auto geolocation + manual fallback) — complete block
# ===============================
if page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Predict Groundwater Potential")

    st.write("### Select values for the predictors:")

    # Ensure session_state key exists to avoid Streamlit runtime errors when writing later
    if "loc_input_fast" not in st.session_state:
        st.session_state["loc_input_fast"] = ""

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
            district = None
    else:
        province = district = None

    st.write("The app will try to fetch your location quickly. Allow location access when prompted, or enter coordinates manually below.")

    # -----------------------------
    # Robust auto geolocation JS
    # -----------------------------
    st.markdown(
        """
        <script>
        function findStreamlitInput() {
            // Try accessing parent document first (for typical Streamlit embed), fall back to local document
            try {
                const parentInputs = Array.from(window.parent.document.querySelectorAll('input'));
                for (const inp of parentInputs) {
                    const aria = inp.getAttribute && inp.getAttribute('aria-label');
                    if ((inp.id && inp.id.includes('loc_input_fast')) || (aria && aria.includes('Hidden location (auto)'))) {
                        return inp;
                    }
                }
            } catch (e) {
                console.warn('[geo] parent.document access blocked or not available:', e);
            }
            const inputsLocal = Array.from(document.querySelectorAll('input'));
            for (const inp of inputsLocal) {
                const aria = inp.getAttribute && inp.getAttribute('aria-label');
                if ((inp.id && inp.id.includes('loc_input_fast')) || (aria && aria.includes('Hidden location (auto)'))) {
                    return inp;
                }
            }
            return null;
        }

        async function tryAutoLocate() {
            console.log('[geo] tryAutoLocate running');
            const inputEl = findStreamlitInput();
            if (!navigator.geolocation) {
                console.warn('[geo] geolocation not available');
                return false;
            }

            return new Promise((resolve) => {
                navigator.geolocation.getCurrentPosition(function(pos) {
                    const val = pos.coords.latitude + ',' + pos.coords.longitude;
                    console.log('[geo] got coords', val);

                    if (inputEl) {
                        try {
                            inputEl.value = val;
                            inputEl.dispatchEvent(new Event('input', { bubbles: true }));
                            inputEl.dispatchEvent(new Event('change', { bubbles: true }));
                            console.log('[geo] wrote coords to input element');
                            resolve(true);
                            return;
                        } catch (e) {
                            console.warn('[geo] writing to input failed', e);
                        }
                    }

                    // Fallbacks
                    try {
                        localStorage.setItem('geo_debug_coords', val);
                        console.log('[geo] wrote coords to localStorage');
                    } catch (e) {
                        console.warn('[geo] localStorage write failed', e);
                    }

                    try {
                        window.parent.postMessage({ type: 'STREAMLIT_GEOCODE', value: val }, '*');
                        console.log('[geo] posted message to parent');
                    } catch (e) {
                        console.warn('[geo] postMessage failed', e);
                    }

                    resolve(true);
                }, function(err) {
                    console.warn('[geo] geolocation error', err);
                    resolve(false);
                }, { enableHighAccuracy: false, maximumAge: 60000, timeout: 7000 });
            });
        }

        // Run soon after load, and expose retry function
        setTimeout(() => {
            tryAutoLocate().then(ok => {
                if (!ok) console.warn('[geo] initial auto-locate failed or was denied');
            });
            window.tryAutoLocateFromStreamlit = tryAutoLocate;
        }, 300);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Hidden input that JS will populate (stable key)
    location = st.text_input("Hidden location (auto)", value=st.session_state.get("loc_input_fast", ""), placeholder="Waiting for browser location...", key="loc_input_fast")

    # Debug expander
    with st.expander("Geolocation debug info (open if not detecting)"):
        st.write("Session state loc_input_fast:", st.session_state.get("loc_input_fast", ""))
        st.write("If nothing appears here after allowing location, open your browser console and look for lines starting with [geo].")
        st.write("You can also run in the console: localStorage.getItem('geo_debug_coords') to see fallback coords if set.")

    # Visible retry button that triggers the JS function exposed on the page
    def trigger_js_retry():
        # No Python action required; re-rendering triggers the small JS below to call the exposed function
        pass

    st.button("Try auto-detect again", on_click=trigger_js_retry)

    # This JS calls the window.tryAutoLocateFromStreamlit function if present (runs after the button causes a rerun)
    st.markdown(
        """
        <script>
        try {
            if (window.tryAutoLocateFromStreamlit && typeof window.tryAutoLocateFromStreamlit === 'function') {
                window.tryAutoLocateFromStreamlit().then(ok => {
                    if (!ok) console.warn('[geo] retry auto-locate failed or was denied');
                });
            }
        } catch (e) {
            console.warn('[geo] retry call error', e);
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Show quick status while waiting
    status = st.empty()
    if not st.session_state.get("loc_input_fast", ""):
        status.info("Fetching location quickly... allow location access when prompted.")
    else:
        status.empty()
        try:
            lat_preview, lon_preview = map(float, st.session_state["loc_input_fast"].split(","))
            st.success(f"Auto-detected: {lat_preview:.6f}, {lon_preview:.6f}")
        except Exception:
            st.error("Auto location couldn't be parsed.")

    # Manual fallback inputs (visible if no auto location)
    def use_manual_location(lat, lon):
        st.session_state["loc_input_fast"] = f"{lat},{lon}"

    manual_lat = manual_lon = None
    if not st.session_state.get("loc_input_fast", ""):
        st.markdown("If automatic detection is slow or denied, enter coordinates manually:")
        c1, c2 = st.columns(2)
        # give manual inputs stable keys to preserve values across reruns
        manual_lat = c1.number_input("Manual Latitude", format="%.6f", value=0.0, key="manual_lat_input")
        manual_lon = c2.number_input("Manual Longitude", format="%.6f", value=0.0, key="manual_lon_input")
        st.button("Use manual location", on_click=use_manual_location, args=(manual_lat, manual_lon))

    # Button to render map for current location (uses the hidden input or manual)
    if st.button("Current Location", key="map-button"):
        loc_val = st.session_state.get("loc_input_fast", "")
        if loc_val:
            try:
                lat, lon = map(float, loc_val.split(","))
                m = folium.Map(location=[lat, lon], zoom_start=15)
                folium.Marker(location=[lat, lon], popup="You are here!", icon=folium.Icon(color='blue')).add_to(m)
                folium_static(m)
            except Exception:
                st.error("Couldn't parse location. Expected 'lat,lon'.")
        else:
            st.error("Location not available. Please allow location access or enter coordinates manually.")

    # User feature inputs
    user_inputs = {}
    for feature in selected_features:
        # If feature exists in data, present choices, otherwise free text
        if feature in data.columns:
            options = sorted(data[feature].dropna().unique().tolist())
            user_inputs[feature] = st.selectbox(f"🔸 {feature}", options, index=0)
        else:
            user_inputs[feature] = st.text_input(f"🔸 {feature}", value=default_values.get(feature, ""))

    # Prediction button
    if st.button("✨ Predict Potential"):
        try:
            full_input = default_values.copy()
            full_input.update(user_inputs)

            # Add numeric lat/lon if available
            loc_val = st.session_state.get("loc_input_fast", "")
            if loc_val:
                try:
                    lat_s, lon_s = loc_val.split(",")
                    full_input['Location_Lat'] = float(lat_s.strip())
                    full_input['Location_Lon'] = float(lon_s.strip())
                except Exception:
                    # if 'Location' is the expected column name instead, preserve raw string
                    full_input['Location'] = loc_val

            # Build input DataFrame with expected full_features columns
            input_df = pd.DataFrame([full_input])[full_features]

            # Encode
            encoded = encoder.transform(input_df)
            encoded_df = pd.DataFrame(encoded, columns=full_features)

            # Select Boruta features (selected_features expected to be subset of full_features post-encoding)
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
