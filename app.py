import streamlit as st
import pandas as pd
import joblib

# ----------- 1. PAGE CONFIGURATION & MODEL LOADING -----------

# Set the page configuration for your app
st.set_page_config(page_title="Exoplanet Hunter", page_icon="🪐", layout="wide")

# Use a caching decorator to load the model only once, improving performance
@st.cache_resource
def load_model():
    """Loads the pre-trained exoplanet classification model."""
    model = joblib.load('model/exoplanet_model.joblib')
    return model

# Load the model
model = load_model()

# ----------- 2. USER INTERFACE -----------

# Set up the title and a description for the app
st.title("Exoplanet Hunter AI 🛰️")
st.write("""
This app uses a Machine Learning model to predict whether a Kepler Object of Interest (KOI) 
is a **CONFIRMED EXOPLANET** or a **FALSE POSITIVE**.
Adjust the sliders in the sidebar to input the physical characteristics of a potential planet candidate.
""")

# Create a sidebar for user inputs
st.sidebar.header("Input Planet Candidate Features")

def user_input_features():
    """Creates sidebar sliders for user to input data."""
    # Note: The default values are chosen to be reasonable averages.
    koi_period = st.sidebar.slider('Orbital Period (days)', 0.1, 500.0, 30.0, 0.1)
    koi_duration = st.sidebar.slider('Transit Duration (hours)', 0.1, 24.0, 3.0, 0.1)
    koi_depth = st.sidebar.slider('Transit Depth (ppm)', 0.0, 200000.0, 1000.0)
    koi_prad = st.sidebar.slider('Planetary Radius (Earth radii)', 0.1, 50.0, 2.5, 0.1)
    koi_steff = st.sidebar.slider('Stellar Temp (K)', 2000.0, 10000.0, 5700.0)
    koi_slogg = st.sidebar.slider('Stellar Surface Gravity (log10(cm/s^2))', 1.0, 6.0, 4.5, 0.1)
    koi_srad = st.sidebar.slider('Stellar Radius (Solar radii)', 0.1, 20.0, 1.0, 0.1)
    
    # Store inputs in a dictionary
    data = {'koi_period': koi_period,
            'koi_duration': koi_duration,
            'koi_depth': koi_depth,
            'koi_prad': koi_prad,
            'koi_steff': koi_steff,
            'koi_slogg': koi_slogg,
            'koi_srad': koi_srad}
    
    # Convert dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user inputs
input_df = user_input_features()

# Display the user-provided inputs
st.subheader('Candidate Features')
st.write(input_df)

# ----------- 3. PREDICTION AND RESULTS -----------

# Add a button to trigger the classification
if st.button('Classify Candidate'):
    # Make prediction using the loaded model
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader('Prediction Result')
    
    # Display the result
    if prediction[0] == 1:
        st.success('**Result: CONFIRMED PLANET** 🪐', icon="✅")
        st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")
    else:
        st.error('**Result: FALSE POSITIVE** ❌', icon="🚨")
        st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")