import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Predictive Maintenance", layout="centered")
st.title("Predictive Maintenance – Engine Condition Classifier")

# Option 1: set HF_MODEL_REPO in Space Secrets (recommended)
# HF Space → Settings → Variables and secrets → add HF_MODEL_REPO = "sastrysagi/capstonemodel"
MODEL_REPO = st.secrets.get("HF_MODEL_REPO", "sastrysagi/capstonemodel")

@st.cache_resource
def load_assets():
    model_file = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.joblib")
    schema_file = hf_hub_download(repo_id=MODEL_REPO, filename="feature_schema.json")
    model = joblib.load(model_file)
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, schema

model, schema = load_assets()

st.write("Enter sensor values and click **Predict**.")
inputs = {}
for col in schema.get("feature_columns", []):
    inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    X = pd.DataFrame([inputs])
    pred = model.predict(X)[0]
    if int(pred) == 1:
        st.error("Prediction: 1 → Maintenance Required")
    else:
        st.success("Prediction: 0 → Normal Operation")
