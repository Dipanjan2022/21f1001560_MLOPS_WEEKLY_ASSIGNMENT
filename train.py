import argparse
import json
import os
import tempfile
from datetime import datetime, timezone

import joblib
import pandas as pd
from google.cloud import storage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = os.path.join(temp_dir, "train.csv")
        eval_path = os.path.join(temp_dir, "eval.csv")

        bucket.blob("data/train.csv").download_to_filename(train_path)
        bucket.blob("data/eval.csv").download_to_filename(eval_path)

        train_df = pd.read_csv(train_path)
        eval_df = pd.read_csv(eval_path)

        target = "species"
        feature_cols = [c for c in train_df.columns if c != target]

        X_train = train_df[feature_cols]
        y_train = train_df[target]
        X_eval = eval_df[feature_cols]
        y_eval = eval_df[target]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ])
        model.fit(X_train, y_train)

        predictions = model.predict(X_eval)
        accuracy = accuracy_score(y_eval, predictions)

        model_path = os.path.join(temp_dir, "model.joblib")
        metrics_path = os.path.join(temp_dir, "metrics.json")
        run_info_path = os.path.join(temp_dir, "run_info.json")

        joblib.dump(model, model_path)

        metrics = {
            "accuracy": float(accuracy),
            "classification_report": classification_report(
                y_eval, predictions, output_dict=True
            ),
            "evaluation_rows": int(len(eval_df)),
            "training_rows": int(len(train_df))
        }

        run_info = {
            "run_id": args.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": "StandardScaler + LogisticRegression",
            "features": feature_cols,
            "target": target
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=2)

        for filename in ["model.joblib", "metrics.json", "run_info.json"]:
            bucket.blob(
                f"artifacts/{args.run_id}/{filename}"
            ).upload_from_filename(os.path.join(temp_dir, filename))
            print(f"Uploaded: artifacts/{args.run_id}/{filename}")

        print(f"\nRun ID: {args.run_id}")
        print(f"Evaluation accuracy: {accuracy:.4f}")
        print("Training completed successfully.")


if __name__ == "__main__":
    main()
