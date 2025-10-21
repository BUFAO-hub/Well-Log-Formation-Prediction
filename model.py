# streamlit_app.py
import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("Well Log Formation Predictor")
st.write("Input well log parameters or upload a CSV to predict formations.")

# -----------------------
# Load sample CSV for memory efficiency
# -----------------------
data_path = r"C:\Users\hp\OneDrive\Documents\taranaki-basin-curated-well-logs\logs.csv"

# Load only first N rows to save memory
try:
    log_sample = pd.read_csv(data_path, nrows=5000)
except MemoryError:
    st.error("Not enough memory to load the dataset. Try reducing nrows.")
    st.stop()

features = ['GR', 'DENS', 'NEUT', 'PEF', 'RESD', 'RESM', 'TEMP', 'DEPT', 'ONSHORE']
log_sample['ONSHORE'] = log_sample['ONSHORE'].astype(int)

# -----------------------
# Load or train model
# -----------------------
def load_or_train_model():
    model_path = "rfc_model.joblib"
    encoder_path = "label_encoder.joblib"

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        st.info("Loaded existing model.")
    else:
        st.warning("Model not found. Training a new model (using sample data)...")

        # Sample for training to reduce memory usage
        train_sample = log_sample.sample(frac=0.5, random_state=42)
        X_train = train_sample[features]
        y_train = train_sample['FORMATION']
        X_train['ONSHORE'] = X_train['ONSHORE'].astype(int)

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y_train)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_encoded)

        # Save model and encoder
        joblib.dump(model, model_path)
        joblib.dump(encoder, encoder_path)
        st.success("Model trained and saved successfully.")

    return model, encoder

model, encoder = load_or_train_model()

# -----------------------
# Single input prediction
# -----------------------
st.subheader("Single Well Prediction")
def user_input_features():
    GR = st.number_input("GR", float(log_sample['GR'].min()), float(log_sample['GR'].max()), float(log_sample['GR'].mean()))
    DENS = st.number_input("DENS", float(log_sample['DENS'].min()), float(log_sample['DENS'].max()), float(log_sample['DENS'].mean()))
    NEUT = st.number_input("NEUT", float(log_sample['NEUT'].min()), float(log_sample['NEUT'].max()), float(log_sample['NEUT'].mean()))
    PEF = st.number_input("PEF", float(log_sample['PEF'].min()), float(log_sample['PEF'].max()), float(log_sample['PEF'].mean()))
    RESD = st.number_input("RESD", float(log_sample['RESD'].min()), float(log_sample['RESD'].max()), float(log_sample['RESD'].mean()))
    RESM = st.number_input("RESM", float(log_sample['RESM'].min()), float(log_sample['RESM'].max()), float(log_sample['RESM'].mean()))
    TEMP = st.number_input("TEMP", float(log_sample['TEMP'].min()), float(log_sample['TEMP'].max()), float(log_sample['TEMP'].mean()))
    DEPT = st.number_input("DEPT", float(log_sample['DEPT'].min()), float(log_sample['DEPT'].max()), float(log_sample['DEPT'].mean()))
    ONSHORE = st.selectbox("ONSHORE", [0, 1])
    data = {
        'GR': GR, 'DENS': DENS, 'NEUT': NEUT, 'PEF': PEF, 'RESD': RESD,
        'RESM': RESM, 'TEMP': TEMP, 'DEPT': DEPT, 'ONSHORE': ONSHORE
    }
    return pd.DataFrame([data])

input_df = user_input_features()

if st.button("Predict Formation"):
    prediction = model.predict(input_df)
    predicted_formation = encoder.inverse_transform(prediction)
    st.success(f"Predicted Formation: {predicted_formation[0]}")

# -----------------------
# Batch CSV prediction
# -----------------------
st.subheader("Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV with the same features", type="csv")

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
    except MemoryError:
        st.error("Not enough memory to read uploaded CSV. Try a smaller file.")
        st.stop()

    if 'ONSHORE' in batch_df.columns:
        batch_df['ONSHORE'] = batch_df['ONSHORE'].astype(int)
    
    if st.button("Predict Batch Formations"):
        batch_pred = model.predict(batch_df[features])
        batch_df['Predicted_Formation'] = encoder.inverse_transform(batch_pred)
        st.write(batch_df)
        st.download_button(
            label="Download Predictions as CSV",
            data=batch_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
