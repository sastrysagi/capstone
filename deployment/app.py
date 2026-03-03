
import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Predictive Maintenance – Engine Condition", layout="centered")

st.title("🚗 Predictive Maintenance – Engine Condition Classifier")
st.caption("Enter sensor values to predict whether the engine needs maintenance (1) or is normal (0).")

MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO", "YOUR_USERNAME/YOUR_MODEL_REPO")

@st.cache_resource
def load_artifacts():
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.joblib")
        schema_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="feature_schema.json")
        model = joblib.load(model_path)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        source = f"HF Hub ({MODEL_REPO_ID})"
        return model, schema, source
    except Exception:
        local_model = os.path.join("models", "best_model.joblib")
        local_schema = os.path.join("models", "feature_schema.json")
        model = joblib.load(local_model)
        with open(local_schema, "r", encoding="utf-8") as f:
            schema = json.load(f)
        source = "Local files (models/)"
        return model, schema, source

model, schema, source = load_artifacts()
st.sidebar.success(f"Model source: {source}")

features = schema["features"]

st.subheader("Sensor Inputs")
cols = st.columns(2)
values = {}
for i, f in enumerate(features):
    with cols[i % 2]:
        values[f] = st.number_input(f, value=0.0, step=0.1, format="%.4f")

input_df = pd.DataFrame([values])
st.write("Input dataframe:")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict"):
    pred = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.error("⚠️ Prediction: MAINTENANCE NEEDED (1)")
    else:
        st.success("✅ Prediction: NORMAL (0)")

    if proba is not None:
        st.metric("Fault probability (class=1)", f"{proba:.3f}")
