import streamlit as st
import pandas as pd
import numpy as np
import joblib 

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
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
selected_features = joblib.load("selected_features.pkl")

# ===============================
# Load Dataset (for categories + full feature list)
# ===============================
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

# ===============================
# PREDICT — safe session_state pattern + postMessage geolocation
# ===============================
if page == "Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("🔮 Predict Groundwater Potential")

    st.write("### Select values for the predictors:")

    # Predefine session_state keys to avoid StreamlitAPIException when assigning later
    if "loc_postmsg" not in st.session_state:
        st.session_state["loc_postmsg"] = ""
    if "manual_lat_input" not in st.session_state:
        st.session_state["manual_lat_input"] = 0.0
    if "manual_lon_input" not in st.session_state:
        st.session_state["manual_lon_input"] = 0.0

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

    # Hidden input populated by client-side JS via postMessage
    loc_input = st.text_input("Hidden location (postMessage)", value=st.session_state.get("loc_postmsg", ""), placeholder="Waiting for location...", key="loc_postmsg")

    # Inject JS (same postMessage approach)
    st.markdown(
        '''
        <script>
        const MSG_TYPE = 'STREAMLIT_POSTMSG_GEO_v1';
        function requestGeolocationAndPost() {
            if (!navigator.geolocation) {
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
                try {
                    if (window.parent && window.parent !== window) {
                        window.parent.postMessage(payload, '*');
                    } else {
                        window.postMessage(payload, '*');
                    }
                } catch (e) {
                    window.postMessage(payload, '*');
                }
            }, function(err){
                window.postMessage({type: MSG_TYPE, ok:false, error: err && err.message ? err.message : 'denied'}, '*');
            }, { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 });
        }
        window.requestGeolocationAndPost = requestGeolocationAndPost;

        window.addEventListener('message', function(event) {
            try {
                const data = event.data;
                if (!data || data.type !== MSG_TYPE) return;
                if (data.ok) {
                    const val = [data.lat, data.lon, data.accuracy !== null ? data.accuracy : '', data.altitude !== null ? data.altitude : '', data.timestamp].join(',');
                    function findInput() {
                        const inputs = Array.from(document.querySelectorAll('input'));
                        for (const inp of inputs) {
                            const aria = inp.getAttribute && inp.getAttribute('aria-label');
                            if ((inp.id && inp.id.indexOf('loc_postmsg') !== -1) || (aria && aria.indexOf('Hidden location (postMessage)') !== -1)) {
                                return inp;
                            }
                        }
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
                        try { localStorage.setItem('streamlit_geo_postmsg', val); } catch(e){}
                        console.warn('[geo-post] no input found; saved to localStorage');
                    }
                } else {
                    console.warn('[geo-post] geolocation failed:', data.error);
                }
            } catch (e) {
                console.error('[geo-post] message handler error', e);
            }
        }, false);

        setTimeout(() => { try { requestGeolocationAndPost(); } catch(e){ console.warn(e); } }, 400);
        </script>
        ''',
        unsafe_allow_html=True,
    )

    # Button to explicitly call the client-side geolocation request
    def _noop(): pass
    st.button("Try auto-detect (postMessage)", on_click=_noop)
    st.markdown(
        """
        <script>
        try {
            if (window.requestGeolocationAndPost && typeof window.requestGeolocationAndPost === 'function') {
                setTimeout(() => { window.requestGeolocationAndPost(); }, 150);
            }
        } catch(e){ console.warn(e); }
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Show status and parsed values
    if not st.session_state.get("loc_postmsg", ""):
        st.info("Waiting for location. Allow location access in your browser when prompted.")
    else:
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
            st.error("Received location couldn't be parsed. Raw: " + str(st.session_state.get("loc_postmsg")))

    # Manual override fields (store values in session_state via widget keys)
    st.markdown("If automatic detection fails, enter coordinates manually:")
    c1, c2 = st.columns(2)
    manual_lat = c1.number_input("Manual Latitude", format="%.6f", value=st.session_state["manual_lat_input"], key="manual_lat_input")
    manual_lon = c2.number_input("Manual Longitude", format="%.6f", value=st.session_state["manual_lon_input"], key="manual_lon_input")

    # Safe callback that receives concrete args and sets session_state
    def _set_manual_loc(lat, lon):
        st.session_state["loc_postmsg"] = f"{lat},{lon},,,"
        st.experimental_rerun()

    st.button("Use manual location", on_click=_set_manual_loc, args=(manual_lat, manual_lon))

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

    st.markdown("---")
    st.write("Feature inputs and prediction UI go here (unchanged).")
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
