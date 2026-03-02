import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Predictive Maintenance – Engine Condition", layout="centered")

# ✅ Use environment variables (HF Space Settings -> Variables/Secrets)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "sastrysagi/capstonemodel")
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional if model repo is public

@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="best_model.joblib",
        repo_type="model",
        token=HF_TOKEN
    )
    schema_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="feature_schema.json",
        repo_type="model",
        token=HF_TOKEN
    )
    model = joblib.load(model_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, schema

try:
    model, schema = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts from '{HF_MODEL_REPO}'. Error: {e}")
    st.stop()

feature_cols = schema.get("feature_columns", [])

st.title("Predictive Maintenance – Engine Condition Classifier")
st.write("Enter sensor values to predict whether the engine needs maintenance (1) or normal (0).")

inputs = {}
for col in feature_cols:
    inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=feature_cols)
    pred = int(model.predict(X)[0])
    st.success(f"Prediction: {pred}  ({'Needs Maintenance' if pred==1 else 'Normal'})")
