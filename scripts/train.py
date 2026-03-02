
import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load dataset
    df = pd.read_csv("data/engine_data.csv")

    target = "Engine_Condition"
    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save splits
    X_train.assign(Engine_Condition=y_train).to_csv("data/train.csv", index=False)
    X_test.assign(Engine_Condition=y_test).to_csv("data/test.csv", index=False)

    # Model + GridSearch
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "best_params": grid.best_params_,
        "created_utc": datetime.utcnow().isoformat() + "Z"
    }

    # Save model
    joblib.dump(best_model, "models/best_model.joblib")

    # Save schema
    schema = {
        "feature_columns": list(X.columns),
        "target_column": target
    }
    with open("models/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Save metrics
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")

if __name__ == "__main__":
    main()
