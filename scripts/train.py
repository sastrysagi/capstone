# scripts/train.py
import os
import json
from datetime import datetime

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def download_raw_csv() -> str:
    """
    Downloads the raw CSV from HF dataset repo.
    Auto-detects a CSV file path by listing repo files.
    Prefers engine_data.csv if present.
    """
    repo_id = os.environ.get("HF_DATASET_REPO")
    token = os.environ.get("HF_TOKEN")

    if not repo_id:
        raise RuntimeError("HF_DATASET_REPO env var is missing. Set it in GitHub Secrets.")

    files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    csv_files = [f for f in files if f.lower().endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in HF dataset repo '{repo_id}'. Files seen: {files[:50]}")

    # Prefer exact match if present
    preferred = None
    for f in csv_files:
        if f.split("/")[-1].lower() == "engine_data.csv":
            preferred = f
            break

    chosen = preferred or csv_files[0]
    print(f"✅ Using raw CSV from HF dataset repo: {repo_id}/{chosen}")

    return hf_hub_download(
        repo_id=repo_id,
        filename=chosen,
        repo_type="dataset",
        token=token,
    )


def basic_clean(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    df = df.drop_duplicates()

    feature_cols = [c for c in df.columns if c != target]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    for c in feature_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


def main():
    ensure_dirs()

    TARGET = "Engine_Condition"

    raw_local_path = download_raw_csv()
    df = pd.read_csv(raw_local_path)

    df = basic_clean(df, TARGET)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    strat = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    # Save splits
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("✅ Saved: data/train.csv and data/test.csv")

    # Model + tuning
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(model, param_grid, scoring="f1", cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
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

    joblib.dump(best_model, "models/best_model.joblib")
    with open("models/feature_schema.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": list(X.columns), "target_column": TARGET}, f, indent=2)
    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Saved model + schema + metrics to models/")
    print("🎉 Training completed successfully.")


if __name__ == "__main__":
    main()
