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
    for f in selected_features:
        st.write(f"• {f}")

    st.markdown("</div>", unsafe_allow_html=True)
# ===============================
# FEATURE GUIDE
# ===============================
elif page == "Feature Guide":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("📘 Feature Guide")

    for col in full_features:
        st.subheader(col)
        st.write("Possible values:")
        st.write(sorted(data[col].dropna().unique().tolist()))

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
