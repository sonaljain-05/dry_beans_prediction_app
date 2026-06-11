import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Config
st.set_page_config(page_title="Dry Bean App", layout="wide")


st.markdown("""
    <style>
    .stApp {
        background-color:#d4fce2;
    }
    </style>
    """, unsafe_allow_html=True)




scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

st.title("🌱 Dry Bean Classification Dashboard")
st.write("Click on the tabs below to adjust features and predict the variety:")
st.write("---")


st.sidebar.header("📋 Bean Information")


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



tab1, tab2, tab3 = st.tabs(["📐 1. Size Features", "📈 2. Shape & Length", "🔬 3. Technical Factors"])


with tab1:
    st.write("### Bean Size Measurements")
    area = st.slider("Area", 20000.0, 250000.0, 60000.0)
    perimeter = st.slider("Perimeter", 500.0, 2000.0, 900.0)
    convex_area = st.slider("Convex Area", 20000.0, 263261.0, 61000.0)


with tab2:
    st.write("### Axis & Structural Dimensions")
    major_axis = st.slider("Major Axis Length", 150.0, 740.0, 300.0)
    minor_axis = st.slider("Minor Axis Length", 100.0, 460.0, 250.0)
    aspect_ratio = st.slider("Aspect Ratio", 1.0, 2.5, 1.5)
    eccentricity = st.slider("Eccentricity", 0.2, 1.0, 0.7)


with tab3:
    st.write("### Microscopic Shape Factors")
    extent = st.slider("Extent", 0.4, 0.9, 0.75)
    solidity = st.slider("Solidity", 0.9, 1.0, 0.98)
    roundness = st.slider("Roundness", 0.4, 1.0, 0.8)
    shape1 = st.slider("Shape Factor 1", 0.002, 0.011, 0.006, step=0.00001, format="%.5f")
    shape2 = st.slider("Shape Factor 2", 0.0005, 0.004, 0.001, step=0.00001, format="%.5f")
