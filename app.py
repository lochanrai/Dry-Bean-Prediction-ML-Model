
import streamlit as st
import joblib
import numpy as np


# 1. Load Model, Scaler and Encoder

model = joblib.load("Dry_Bean_Model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# 2. App Heading
st.title(" Dry Bean Species Predictor ")

st.image(
    "https://www.almanac.com/sites/default/files/images/beans-shutterstock_747509755.jpg",
    use_container_width=True,
    caption="Discover the different varieties of dry beans"
)

st.write(
    "Welcome! Enter the physical measurements of a dry bean "
    "to predict its species using Machine Learning."
)

# --------------------------------
# 4. Input Features
# --------------------------------

st.subheader("📏 Bean Measurements")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area", min_value=0.0, max_value=10000.0, value=0.0)
    perimeter = st.number_input("Perimeter", min_value=0.0, max_value=1000.0, value=0.0)
    major_axis_length = st.number_input("Major Axis Length", min_value=0.0, max_value=100.0, value=0.0)
    minor_axis_length = st.number_input("Minor Axis Length", min_value=0.0, max_value=100.0, value=0.0)
    aspect_ratio = st.slider("Aspect Ratio", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    eccentricity = st.slider("Eccentricity", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    convex_area = st.slider("Convex Area", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
    equivalent_diameter = st.slider("Equivalent Diameter", min_value=0.0, max_value=100.0, value=0.0)

with col2:
    extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    solidity = st.number_input("Solidity", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    roundness = st.slider("Roundness", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    compactness = st.slider("Compactness", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    shape_factor1 = st.slider("Shape Factor 1", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
    shape_factor2 = st.slider("Shape Factor 2", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
    shape_factor3 = st.slider("Shape Factor 3", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    shape_factor4 = st.slider("Shape Factor 4", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")

st.divider()

# --------------------------------
# 5. Prediction
# --------------------------------

if st.button("🔍 Predict Bean Species", use_container_width=True):

    # Collect input values
    data = np.array([[
        area, perimeter, major_axis_length, minor_axis_length,
        aspect_ratio, eccentricity, convex_area, equivalent_diameter,
        extent, solidity, roundness, compactness,
        shape_factor1, shape_factor2, shape_factor3, shape_factor4
    ]])

    # Scale input data
    data_scaled = scaler.transform(data)

    # Predict species
    # Predict species
    prediction = model.predict(data_scaled)

    # Convert encoded prediction into original category name
    predicted_species = label_encoder.inverse_transform(prediction)

    st.subheader("Predicted Dry Bean Species")
    st.markdown(f"## {predicted_species[0]}")

    