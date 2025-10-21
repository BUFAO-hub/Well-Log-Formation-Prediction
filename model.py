import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load pre-trained models
# -----------------------
model = joblib.load("rfc_model.joblib")
encoder = joblib.load("label_encoder.joblib")

# -----------------------
# Streamlit UI
# -----------------------
st.title("Well Log Formation Predictor")
st.write("Input well log parameters to predict the formation.")

def user_input_features():
    GR = st.number_input("GR", 0.0, 150.0, 50.0)
    DENS = st.number_input("DENS", 1.0, 3.0, 2.0)
    NEUT = st.number_input("NEUT", 0.0, 10.0, 5.0)
    PEF = st.number_input("PEF", 1.0, 10.0, 5.0)
    RESD = st.number_input("RESD", 0.0, 100.0, 50.0)
    RESM = st.number_input("RESM", 0.0, 100.0, 50.0)
    TEMP = st.number_input("TEMP", 20.0, 200.0, 100.0)
    DEPT = st.number_input("DEPT", 0.0, 10000.0, 1000.0)
    ONSHORE = st.selectbox("ONSHORE", [0, 1])
    return pd.DataFrame([{
        'GR': GR, 'DENS': DENS, 'NEUT': NEUT, 'PEF': PEF,
        'RESD': RESD, 'RESM': RESM, 'TEMP': TEMP, 'DEPT': DEPT,
        'ONSHORE': ONSHORE
    }])

input_df = user_input_features()

if st.button("Predict Formation"):
    prediction = model.predict(input_df)
    predicted_formation = encoder.inverse_transform(prediction)
    st.subheader("Predicted Formation")
    st.write(predicted_formation[0])
