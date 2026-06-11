import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Config
st.set_page_config(page_title="Dry Bean App", layout="wide")

# 🔥 Super Strong CSS to FORCE Tabs and headers to highlight beautifully
st.markdown("""
    <style>
    /* Premium Soft Background */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* Force Streamlit Tabs to be Huge, Bold, and Colorful */
    button[data-baseweb="tab"] {
        background-color: #e8f5e9 !important; /* Light Green background for tab buttons */
        border-radius: 10px 10px 0px 0px !important;
        padding: 12px 24px !important;
        margin-right: 5px !important;
        border: 1px solid #c8e6c9 !important;
    }
    
    /* Active Tab Style */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2e7d32 !important; /* Dark Green when selected */
    }
    
    /* Tab Text styling */
    button[data-baseweb="tab"] p {
        font-size: 22px !important; /* Huge Font Size */
        font-weight: 900 !important; /* Extra Bold */
        color: #1b5e20 !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: white !important; /* White text for active tab */
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model
scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

# Main Header
st.markdown("<h1 style='color: #1b5e20;'>🌱 Dry Bean Classification Dashboard</h1>", unsafe_allow_html=True)
st.write("Click on the massive highlighted tabs below to adjust features:")
st.write("---")

# 3. SIDEBAR: Image + Target Varieties List
st.sidebar.header("📋 Bean Information")

# Live Green Beans Image URL
st.sidebar.image(
    "beans.jpg", 
    caption="Fresh Green Beans Analysis", 
    use_container_width=True
)

st.sidebar.subheader("🎯 Target Varieties:")
st.sidebar.markdown("""
- **BARBUNYA**
- **BOMBAY**
- **CALI**
- **DERMASON**
- **HOROZ**
- **SEKER**
- **SIRA**
""")

st.sidebar.write("---")
st.sidebar.markdown("""
### How to use:
1. **Switch Tabs:** Go through different tabs to set bean details.
2. **Predict:** Click the big button at the bottom.
""")
st.sidebar.write("---")
st.sidebar.success("🤖 Model Status: Loaded & Ready")


# 4. TABS LAYOUT (Super Highlighted Now)
tab1, tab2, tab3 = st.tabs(["📐 SIZE FEATURES", "📈 SHAPE & LENGTH", "🔬 TECHNICAL FACTORS"])

# Tab 1: Size features
with tab1:
    # Double Highlighted Internal Header using HTML
    st.markdown("<div style='background-color: #2e7d32; padding: 10px; border-radius: 5px;'><h2 style='color: white; margin: 0; padding-left: 10px;'>🟢 SECTION 1: BEAN SIZE MEASUREMENTS</h2></div>", unsafe_allow_html=True)
    st.write("") # Blank space
    area = st.slider("Area", 20000.0, 250000.0, 60000.0)
    perimeter = st.slider("Perimeter", 500.0, 2000.0, 900.0)
    convex_area = st.slider("Convex Area", 20000.0, 263261.0, 61000.0)

# Tab 2: Shape and Length features
with tab2:
    # Double Highlighted Internal Header using HTML
    st.markdown("<div style='background-color: #1565c0; padding: 10px; border-radius: 5px;'><h2 style='color: white; margin: 0; padding-left: 10px;'>🔵 SECTION 2: AXIS, SHAPE & LENGTH</h2></div>", unsafe_allow_html=True)
    st.write("") # Blank space
    major_axis = st.slider("Major Axis Length", 150.0, 740.0, 300.0)
    minor_axis = st.slider("Minor Axis Length", 100.0, 460.0, 250.0)
    aspect_ratio = st.slider("Aspect Ratio", 1.0, 2.5, 1.5)
    eccentricity = st.slider("Eccentricity", 0.2, 1.0, 0.7)

# Tab 3: Technical Shape Factors
with tab3:
    # Double Highlighted Internal Header using HTML
    st.markdown("<div style='background-color: #e65100; padding: 10px; border-radius: 5px;'><h2 style='color: white; margin: 0; padding-left: 10px;'>🟠 SECTION 3: MICROSCOPIC SHAPE FACTORS</h2></div>", unsafe_allow_html=True)
    st.write("") # Blank space
    extent = st.slider("Extent", 0.4, 0.9, 0.75)
    solidity = st.slider("Solidity", 0.9, 1.0, 0.98)
    roundness = st.slider("Roundness", 0.4, 1.0, 0.8)
    shape1 = st.slider("Shape Factor 1", 0.002, 0.011, 0.006, step=0.00001, format="%.5f")
    shape2 = st.slider("Shape Factor 2", 0.0005, 0.004, 0.001, step=0.00001, format="%.5f")
    shape3 = st.slider("Shape Factor 3", 0.4, 1.0, 0.6)

st.write("---")


# 5. Predict Button and Result Display
if st.button("🚀 Predict Bean Variety", use_container_width=True):
    
    # Data Preparation
    input_dict = {
        "area": area,
        "perimeter": perimeter,
        "majoraxislength": major_axis,
        "minoraxislength": minor_axis,
        "aspectration": aspect_ratio,  
        "eccentricity": eccentricity,
        "convexarea": convex_area,
        "extent": extent,
        "solidity": solidity,
        "roundness": roundness,
        "shapefactor1": shape1,
        "shapefactor2": shape2,
        "shapefactor3": shape3
    }
    input_df = pd.DataFrame([input_dict])

    # Log Transformation
    input_df['area'] = np.log1p(input_df['area'])
    input_df['convexarea'] = np.log1p(input_df['convexarea'])
    input_df['minoraxislength'] = np.log1p(input_df['minoraxislength'])
    input_df['perimeter'] = np.log1p(input_df['perimeter'])
    input_df['majoraxislength'] = np.log1p(input_df['majoraxislength'])

    # Scale & Predict
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)

    # Class Names Mapping
    class_mapping = {0: "BARBUNYA", 1: "BOMBAY", 2: "CALI", 3: "DERMASON", 4: "HOROZ", 5: "SEKER", 6: "SIRA"}
    pred_val = prediction[0]
    final_class = class_mapping.get(pred_val, pred_val)
    
    # Metric Card output
    st.write("### 🎯 Result:")
    st.metric(label="Predicted Botanical Variety", value=f"✨ {final_class} ✨")
