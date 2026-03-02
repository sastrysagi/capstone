import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")
HF_SPACE_REPO = os.environ.get("HF_SPACE_REPO")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing. Add it as a GitHub Actions secret.")
if not HF_DATASET_REPO:
    raise RuntimeError("HF_DATASET_REPO is missing. Add it as a GitHub Actions secret.")
if not HF_MODEL_REPO:
    raise RuntimeError("HF_MODEL_REPO is missing. Add it as a GitHub Actions secret.")
if not HF_SPACE_REPO:
    raise RuntimeError("HF_SPACE_REPO is missing. Add it as a GitHub Actions secret.")

api = HfApi(token=HF_TOKEN)

# ----- Files produced by the notebook -----
train_path = "data/train.csv"
test_path = "data/test.csv"

model_path = "models/best_model.joblib"
schema_path = "models/feature_schema.json"
model_card_path = "models/README.md"

space_folder = "deployment"  # app.py + requirements.txt (+ optional assets)

# ----- Upload Dataset splits -----
api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
api.upload_file(path_or_fileobj=train_path, path_in_repo="processed/train.csv",
                repo_id=HF_DATASET_REPO, repo_type="dataset",
                commit_message="Add processed train split")
api.upload_file(path_or_fileobj=test_path, path_in_repo="processed/test.csv",
                repo_id=HF_DATASET_REPO, repo_type="dataset",
                commit_message="Add processed test split")
print("✅ Uploaded dataset splits to:", HF_DATASET_REPO)

# ----- Upload Model artifacts -----
api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=model_path, path_in_repo="best_model.joblib",
                repo_id=HF_MODEL_REPO, repo_type="model",
                commit_message="Upload best model")
api.upload_file(path_or_fileobj=schema_path, path_in_repo="feature_schema.json",
                repo_id=HF_MODEL_REPO, repo_type="model",
                commit_message="Upload feature schema")

# Model card is optional but recommended
if os.path.exists(model_card_path):
    api.upload_file(path_or_fileobj=model_card_path, path_in_repo="README.md",
                    repo_id=HF_MODEL_REPO, repo_type="model",
                    commit_message="Add model card")

print("✅ Uploaded model artifacts to:", HF_MODEL_REPO)

# ----- Upload HF Space (Streamlit) -----
api.create_repo(repo_id=HF_SPACE_REPO, repo_type="space", exist_ok=True, space_sdk="streamlit")
api.upload_folder(folder_path=space_folder, repo_id=HF_SPACE_REPO, repo_type="space",
                  commit_message="Automated Space update from GitHub Actions")

print("✅ Uploaded Space app to:", HF_SPACE_REPO)
