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
# PREDICT — postMessage geolocation method (replace your Predict block with this)
# ===============================
if page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Predict Groundwater Potential")

    st.write("### Select values for the predictors:")

    # Ensure session_state key exists
    if "loc_postmsg" not in st.session_state:
        st.session_state["loc_postmsg"] = ""  # will hold "lat,lon,accuracy,timestamp" or empty

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

    st.write("This version uses a safer postMessage flow to get location from the browser. Allow location when prompted, then click 'Try auto-detect' if needed.")

    # Hidden input that will be populated by the JS message listener
    loc_input = st.text_input("Hidden location (postMessage)", value=st.session_state.get("loc_postmsg", ""), placeholder="Waiting for location...", key="loc_postmsg")

    # Inject JS that both requests location (when asked) and listens for incoming postMessage,
    # then writes received coords into the hidden input.
    st.markdown(
        """
        <script>
        // Unique message type so we don't collide with other pages
        const MSG_TYPE = 'STREAMLIT_POSTMSG_GEO_v1';

        // Function to request geolocation from the browser and post it to this window (or parent)
        function requestGeolocationAndPost() {
            if (!navigator.geolocation) {
                console.warn('[geo-post] navigator.geolocation not available');
                window.postMessage({type: MSG_TYPE, ok:false, error: 'no_geolocation'}, '*');
                return;
            }
            navigator.geolocation.getCurrentPosition(function(pos) {
                const payload = {
                    type: MSG_TYPE,
                    ok: true,
                    lat: pos.coords.latitude,
                    lon: pos.coords.longitude,
                    accuracy: pos.coords.accuracy || null,
                    altitude: pos.coords.altitude || null,
                    timestamp: pos.timestamp || Date.now()
                };
                // If we're embedded, post to parent; if top-level, post to window
                try {
                    if (window.parent && window.parent !== window) {
                        window.parent.postMessage(payload, '*');
                    } else {
                        window.postMessage(payload, '*');
                    }
                } catch (e) {
                    // as fallback, post to window
                    window.postMessage(payload, '*');
                }
            }, function(err){
                console.warn('[geo-post] geolocation error', err);
                window.postMessage({type: MSG_TYPE, ok:false, error: err && err.message ? err.message : 'denied'}, '*');
            }, { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 });
        }

        // Expose the function so Streamlit-triggered reruns can call it
        window.requestGeolocationAndPost = requestGeolocationAndPost;

        // Listener: receive the geo message and write into the Streamlit input field safely
        window.addEventListener('message', function(event) {
            try {
                const data = event.data;
                if (!data || data.type !== MSG_TYPE) return;
                // Compose a compact value: lat,lon,accuracy,timestamp
                if (data.ok) {
                    const val = [data.lat, data.lon, data.accuracy !== null ? data.accuracy : '', data.altitude !== null ? data.altitude : '', data.timestamp].join(',');
                    // Try to find the Streamlit input element by key/label heuristics
                    function findInput() {
                        const inputs = Array.from(document.querySelectorAll('input'));
                        for (const inp of inputs) {
                            const aria = inp.getAttribute && inp.getAttribute('aria-label');
                            if ((inp.id && inp.id.indexOf('loc_postmsg') !== -1) || (aria && aria.indexOf('Hidden location (postMessage)') !== -1) || (aria && aria.indexOf('Hidden location (postMessage)') !== -1)) {
                                return inp;
                            }
                        }
                        // fallback: choose the first visible text input
                        return inputs.find(i => i.type === 'text' || i.type === 'search') || null;
                    }
                    const inputEl = findInput();
                    if (inputEl) {
                        inputEl.focus();
                        inputEl.value = val;
                        inputEl.dispatchEvent(new Event('input', { bubbles: true }));
                        inputEl.dispatchEvent(new Event('change', { bubbles: true }));
                        console.log('[geo-post] wrote coords to input:', val);
                    } else {
                        // As a fallback write to localStorage so the user can copy it
                        try { localStorage.setItem('streamlit_geo_postmsg', val); } catch(e){}
                        console.warn('[geo-post] could not find input element; saved to localStorage');
                    }
                } else {
                    console.warn('[geo-post] geolocation failed:', data.error);
                }
            } catch (e) {
                console.error('[geo-post] message handler error', e);
            }
        }, false);

        // Also attempt to auto-run once after load (gives permission prompt immediately)
        setTimeout(() => {
            try {
                requestGeolocationAndPost();
            } catch(e) {
                console.warn('[geo-post] auto-run failed', e);
            }
        }, 400);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Button to explicitly trigger the geolocation request (useful after a Streamlit rerun)
    def trigger_postmsg_geo():
        # no Python-side action; the small JS below will call the exposed function in the page
        pass

    st.button("Try auto-detect (postMessage)", on_click=trigger_postmsg_geo)

    # Small JS snippet to call the exposed function after a rerun (so the button triggers the client-side function)
    st.markdown(
        """
        <script>
        try {
            if (window.requestGeolocationAndPost && typeof window.requestGeolocationAndPost === 'function') {
                // delay slightly to allow function to exist
                setTimeout(() => { window.requestGeolocationAndPost(); }, 150);
            }
        } catch(e){
            console.warn('[geo-post] call after rerun failed', e);
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Show status and parsed values
    if not st.session_state.get("loc_postmsg", ""):
        st.info("Waiting for location. Allow location access in your browser when prompted.")
    else:
        # Attempt to parse stored value: lat,lon,accuracy,alt,timestamp
        try:
            parts = st.session_state["loc_postmsg"].split(",")
            lat = float(parts[0])
            lon = float(parts[1])
            acc = parts[2] if len(parts) > 2 and parts[2] != '' else None
            alt = parts[3] if len(parts) > 3 and parts[3] != '' else None
            ts = parts[4] if len(parts) > 4 else None
            st.success(f"Auto-detected: {lat:.6f}, {lon:.6f}")
            if acc:
                st.write(f"Accuracy: {acc} m")
            if alt:
                st.write(f"Altitude: {alt} m")
            if ts:
                try:
                    import datetime
                    t = datetime.datetime.fromtimestamp(float(ts)/1000.0) if float(ts) > 1e12 else datetime.datetime.fromtimestamp(float(ts))
                    st.write(f"Timestamp: {t.isoformat()}")
                except Exception:
                    st.write("Timestamp:", ts)
        except Exception:
            st.error("Received location couldn't be parsed. Raw:", st.session_state.get("loc_postmsg"))

    # Manual override fields
    st.markdown("If automatic detection fails, enter coordinates manually:")
    c1, c2 = st.columns(2)
    manual_lat = c1.number_input("Manual Latitude", format="%.6f", value=0.0, key="manual_lat_input")
    manual_lon = c2.number_input("Manual Longitude", format="%.6f", value=0.0, key="manual_lon_input")
    if st.button("Use manual location"):
        st.session_state["loc_postmsg"] = f"{manual_lat},{manual_lon},,,"
        st.experimental_rerun()

    # Map preview button (uses current loc_postmsg)
    if st.button("Show map for current location"):
        loc_val = st.session_state.get("loc_postmsg", "")
        if loc_val:
            try:
                lat_s, lon_s = loc_val.split(",")[0:2]
                latf = float(lat_s); lonf = float(lon_s)
                m = folium.Map(location=[latf, lonf], zoom_start=15)
                folium.Marker(location=[latf, lonf], popup="Current location", icon=folium.Icon(color='blue')).add_to(m)
                folium_static(m)
            except Exception:
                st.error("Couldn't parse stored coordinates.")
        else:
            st.error("No coordinates stored. Try auto-detect or enter manually.")

    # Continue with the rest of your Predict logic (feature inputs, model prediction, etc.)
    st.markdown("---")
    st.write("Feature inputs and prediction UI go here (unchanged).")
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
