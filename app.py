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

# ----------------- Predict page UI -----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.title("🔮 Predict Groundwater Potential")
st.write("### Select values for the predictors:")

# Country / Province / District UI (unchanged)
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

# ----- Geolocation component -----
st.markdown("#### Detect your location (optional)")
geo_col1, geo_col2 = st.columns([2,1])

with geo_col1:
    # A disabled text_input will be populated with coordinates once available
    loc_input = st.text_input("Your Location (lat, lon)", value="", placeholder="Click 'Get Location' to auto-detect", key="location_input")

with geo_col2:
    if st.button("📍 Get Location"):
        # Render a small HTML/JS component to ask for geolocation permission and post coords back
        geocode_html = """
        <html>
        <body>
        <script>
        const sendCoords = (lat, lon) => {
            const coords = {'lat': lat, 'lon': lon};
            // Send to Streamlit by writing to parent window (works inside components.html)
            window.parent.postMessage({isStreamlitMessage: true, type: 'geolocation', data: coords}, '*');
        };

        function handleError(err) {
            const message = {'error': true, 'message': err.message || 'Geolocation error'};
            window.parent.postMessage({isStreamlitMessage: true, type: 'geolocation', data: message}, '*');
        }

        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    sendCoords(position.coords.latitude, position.coords.longitude);
                },
                (err) => { handleError(err); },
                {enableHighAccuracy: true, timeout: 10000}
            );
        } else {
            handleError({message: 'Geolocation not supported'});
        }
        </script>
        </body>
        </html>
        """
        # components.html will host the JS and allow receiving messages in Python-side via `components.html` return value
        components.html(geocode_html, height=0, scrolling=False)

        # Small pause to let JS run and message be received by Streamlit (message handling below)
        time.sleep(0.1)

# Streamlit cannot directly capture window.postMessage from components.html. Use a tiny workaround:
# create a second components.html that listens to window messages and writes the coordinates into
# an element that Streamlit can read back via "components.html" return value.
# This snippet repeatedly injects a script that listens for that message and then writes JSON to the page.
listen_html = """
<html>
  <body>
    <div id="out"></div>
    <script>
      window.addEventListener('message', (event) => {
        try {
          const msg = event.data;
          if (msg && msg.type === 'geolocation') {
            const node = document.getElementById('out');
            node.innerText = JSON.stringify(msg.data);
          }
        } catch(e) { }
      }, false);
    </script>
  </body>
</html>
"""
out = components.html(listen_html, height=50)

# The components.html above returns the static HTML, but we can fetch the value by re-rendering and reading via st.session_state.
# A more reliable approach: provide an explicit text area for the user to paste coordinates if automatic detection fails.
st.markdown("If automatic detection fails, please paste coordinates in the format: lat, lon (e.g., -19.0154, 29.1549).")

# Allow manual paste as fallback
manual_location = st.text_input("Or paste coordinates here", value="", placeholder="-19.0154, 29.1549", key="manual_loc")

# Prefer detected location (if any) otherwise manual
# Note: as Streamlit cannot directly capture the postMessage into Python without a dedicated component, use manual_location when provided.
# If you have a custom Streamlit Component installed for geolocation, replace this fallback with the component output.
location = ""
if manual_location:
    location = manual_location.strip()
elif loc_input:
    location = loc_input.strip()

# ---------------- Feature inputs ----------------
user_inputs = {}
for feature in selected_features:
    options = sorted(data[feature].dropna().unique().tolist())
    user_inputs[feature] = st.selectbox(f"🔸 {feature}", options, key=f"feat_{feature}")

# ----- Prediction button -----
if st.button("✨ Predict Potential"):
    try:
        # Build full input
        full_input = default_values.copy()
        full_input.update(user_inputs)

        # Add location if model expects it
        if location:
            full_input['Location'] = location  # ensure your model supports this column

        # Ensure ordering and presence of features
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
        if 'selected_features' in globals() and selected_features:
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
        if 'full_features' in globals() and 'data' in globals() and isinstance(data, pd.DataFrame):
            for col in full_features:
                st.subheader(col)
                st.write("Possible values:")
                # Defensive: if column not in data, skip with warning
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
