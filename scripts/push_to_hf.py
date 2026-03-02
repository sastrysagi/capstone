import os
import json
from pathlib import Path

from huggingface_hub import HfApi

# --------- ENV (from GitHub Actions Secrets) ----------
HF_TOKEN        = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")  # e.g. "sastrysagi/capstonedataset"
HF_MODEL_REPO   = os.environ.get("HF_MODEL_REPO")    # e.g. "sastrysagi/capstonemodel"
HF_SPACE_REPO   = os.environ.get("HF_SPACE_REPO")    # e.g. "sastrysagi/capstonespace"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing. Add it as a GitHub Secret.")
if not HF_DATASET_REPO:
    raise RuntimeError("HF_DATASET_REPO is missing. Add it as a GitHub Secret.")
if not HF_MODEL_REPO:
    raise RuntimeError("HF_MODEL_REPO is missing. Add it as a GitHub Secret.")
if not HF_SPACE_REPO:
    raise RuntimeError("HF_SPACE_REPO is missing. Add it as a GitHub Secret.")

api = HfApi(token=HF_TOKEN)

ROOT = Path(".")
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DEPLOY_DIR = ROOT / "deployment"


def ensure_file(path: Path, message: str):
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path.resolve()}")


def write_deployment_files():
    """
    Create docker-based Space files for Streamlit deployment.
    HF Spaces now expects sdk: docker / gradio / static for repo creation.
    Streamlit apps should be deployed using Docker template.
    """
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # 1) requirements.txt (include what your app needs)
    req_path = DEPLOY_DIR / "requirements.txt"
    if not req_path.exists():
        req_path.write_text(
            "\n".join([
                "streamlit>=1.30.0",
                "pandas",
                "numpy",
                "scikit-learn",
                "joblib",
                "huggingface_hub",
            ]) + "\n",
            encoding="utf-8"
        )

    # 2) app.py (loads model from HF model repo at runtime)
    app_path = DEPLOY_DIR / "app.py"
    if not app_path.exists():
        app_path.write_text(
            r'''
import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Predictive Maintenance – Engine Condition", layout="centered")

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_MODEL_REPO:
    st.error("HF_MODEL_REPO env var missing. Add it to Space Secrets.")
    st.stop()

@st.cache_resource
def load_artifacts():
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib", repo_type="model", token=HF_TOKEN)
    schema_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="feature_schema.json", repo_type="model", token=HF_TOKEN)
    model = joblib.load(model_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, schema

model, schema = load_artifacts()
feature_cols = schema["feature_columns"]

st.title("Predictive Maintenance – Engine Condition Classifier")

st.write("Enter sensor values to predict whether the engine needs maintenance (1) or normal (0).")

inputs = {}
for col in feature_cols:
    inputs[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=feature_cols)
    pred = int(model.predict(X)[0])
    st.success(f"Prediction: {pred}  ({'Needs Maintenance' if pred==1 else 'Normal'})")
'''.strip() + "\n",
            encoding="utf-8"
        )

    # 3) Dockerfile (streamlit served on 7860 for HF docker spaces)
    docker_path = DEPLOY_DIR / "Dockerfile"
    docker_path.write_text(
        r'''
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
'''.strip() + "\n",
        encoding="utf-8"
    )

    # 4) README.md with YAML config for docker spaces
    readme_path = DEPLOY_DIR / "README.md"
    readme_path.write_text(
        f"""---
title: Predictive Maintenance – Engine Condition
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Predictive Maintenance – Engine Condition Classifier

This Space serves a Streamlit UI that loads the trained model from the Hugging Face **Model Hub**:

- Model repo: `{HF_MODEL_REPO}`
""",
        encoding="utf-8"
    )


def upload_dataset_splits():
    ensure_file(DATA_DIR / "train.csv", "Train split missing (train.py should create it)")
    ensure_file(DATA_DIR / "test.csv", "Test split missing (train.py should create it)")

    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(DATA_DIR / "train.csv"),
        path_in_repo="processed/train.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="Add processed train split"
    )
    api.upload_file(
        path_or_fileobj=str(DATA_DIR / "test.csv"),
        path_in_repo="processed/test.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="Add processed test split"
    )
    print("✅ Uploaded dataset splits to:", HF_DATASET_REPO)


def upload_model_artifacts():
    ensure_file(MODELS_DIR / "best_model.joblib", "Model missing (train.py should create it)")
    ensure_file(MODELS_DIR / "feature_schema.json", "Schema missing (train.py should create it)")

    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(MODELS_DIR / "best_model.joblib"),
        path_in_repo="best_model.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message="Upload best model"
    )
    api.upload_file(
        path_or_fileobj=str(MODELS_DIR / "feature_schema.json"),
        path_in_repo="feature_schema.json",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message="Upload feature schema"
    )

    # Optional: upload metrics if present
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        api.upload_file(
            path_or_fileobj=str(metrics_path),
            path_in_repo="metrics.json",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            commit_message="Upload metrics"
        )

    # Minimal model card
    model_readme = MODELS_DIR / "README.md"
    model_readme.write_text(
        f"# Predictive Maintenance – Engine Condition Classifier\n\n"
        f"Artifacts:\n"
        f"- `best_model.joblib`\n"
        f"- `feature_schema.json`\n",
        encoding="utf-8"
    )
    api.upload_file(
        path_or_fileobj=str(model_readme),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message="Add model card"
    )

    print("✅ Uploaded model artifacts to:", HF_MODEL_REPO)


def create_and_upload_space():
    # IMPORTANT FIX: Streamlit SDK is no longer accepted; use docker.
    # HF expects sdk: gradio|docker|static for create_repo now.
    write_deployment_files()

    api.create_repo(repo_id=HF_SPACE_REPO, repo_type="space", exist_ok=True, space_sdk="docker")

    # Upload deployment folder contents to Space root
    for fname in ["app.py", "requirements.txt", "Dockerfile", "README.md"]:
        fpath = DEPLOY_DIR / fname
        ensure_file(fpath, "Deployment file missing")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,          # root of space repo
            repo_id=HF_SPACE_REPO,
            repo_type="space",
            commit_message=f"Update Space file: {fname}"
        )

    print("✅ Uploaded Space (docker) files to:", HF_SPACE_REPO)


if __name__ == "__main__":
    upload_dataset_splits()
    upload_model_artifacts()
    create_and_upload_space()
