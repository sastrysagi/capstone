# scripts/train.py
"""
Capstone3 – Predictive Maintenance (CI-friendly training script)

Fixes your GitHub Actions error:
- Does NOT assume data/engine_data.csv exists in the repo.
- Downloads raw CSV from Hugging Face Dataset repo via hf_hub_download.

Expected:
- HF_DATASET_REPO secret/env set to: username/dataset_repo (e.g., sastrysagi/capstonedataset)
- HF_TOKEN secret/env set (required if dataset is private; ok if public too)
- Raw file exists in the HF dataset repo at one of these paths:
    1) engine_data.csv
    2) data/engine_data.csv
    3) raw/engine_data.csv

Outputs (created in repo workspace):
- data/train.csv
- data/test.csv
- models/best_model.joblib
- models/feature_schema.json
- models/metrics.json
"""

import os
import json
from datetime import datetime

import joblib
import pandas as pd

from huggingface_hub import hf_hub_download

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def download_raw_csv() -> str:
    """Try a few common filenames in the HF dataset repo."""
    repo_id = os.environ.get("HF_DATASET_REPO")
    token = os.environ.get("HF_TOKEN")  # optional for public datasets

    if not repo_id:
        raise RuntimeError("HF_DATASET_REPO env var is missing. Set it in GitHub Secrets.")

    candidate_filenames = [
        "engine_data.csv",
        "data/engine_data.csv",
        "raw/engine_data.csv",
    ]

    last_err = None
    for fname in candidate_filenames:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                token=token,
            )
            print(f"✅ Downloaded raw dataset from HF: {repo_id}/{fname}")
            return local_path
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        f"Could not find raw CSV in HF dataset repo '{repo_id}'.\n"
        f"Tried: {candidate_filenames}\n"
        f"Last error: {last_err}"
    )


def basic_clean(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    df = df.drop_duplicates()

    # Coerce features to numeric
    feature_cols = [c for c in df.columns if c != target]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce target to numeric then int
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    # Fill missing values in features with median
    for c in feature_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


def main():
    ensure_dirs()

    TARGET = "Engine_Condition"

    # 1) Load dataset from HF (not from repo filesystem)
    raw_local = download_raw_csv()
    df = pd.read_csv(raw_local)

    # 2) Clean
    df = basic_clean(df, TARGET)

    # 3) Split
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratify if both classes present
    strat = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    # Save splits
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values

    train_path = "data/train.csv"
    test_path = "data/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"✅ Saved splits: {train_path}, {test_path}")

    # 4) Model + tuning
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("✅ Best params:", grid.best_params_)
    print("✅ Best CV F1:", float(grid.best_score_))

    # 5) Evaluate
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "best_params": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    print("✅ Test metrics:", {k: v for k, v in metrics.items() if k not in ["best_params"]})

    # 6) Save model
    model_path = "models/best_model.joblib"
    joblib.dump(best_model, model_path)
    print("✅ Saved model:", model_path)

    # 7) Save schema
    schema = {
        "feature_columns": list(X.columns),
        "target_column": TARGET,
        "raw_source": "huggingface_dataset",
        "raw_file_downloaded_to": raw_local,
    }
    schema_path = "models/feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print("✅ Saved schema:", schema_path)

    # 8) Save metrics
    metrics_path = "models/metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved metrics:", metrics_path)

    print("\n🎉 Training completed successfully.")


if __name__ == "__main__":
    main()
