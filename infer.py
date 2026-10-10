import argparse
import json
import os
import tempfile
from datetime import datetime, timezone

import joblib
import pandas as pd
from google.cloud import storage
from sklearn.metrics import accuracy_score, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    prefix = f"artifacts/{args.run_id}"

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.joblib")
        eval_path = os.path.join(temp_dir, "eval.csv")
        predictions_path = os.path.join(temp_dir, "predictions.csv")
        metrics_path = os.path.join(temp_dir, "inference_metrics.json")

        bucket.blob(
            f"{prefix}/model.joblib"
        ).download_to_filename(model_path)

        bucket.blob("data/eval.csv").download_to_filename(eval_path)

        model = joblib.load(model_path)
        eval_df = pd.read_csv(eval_path)

        target = "species"
        features = [c for c in eval_df.columns if c != target]

        X_eval = eval_df[features]
        y_true = eval_df[target]
        y_pred = model.predict(X_eval)

        results = pd.DataFrame({
            "actual_species": y_true,
            "predicted_species": y_pred,
            "correct": y_true == y_pred
        })
        results.to_csv(predictions_path, index=False)

        metrics = {
            "run_id": args.run_id,
            "inference_timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "evaluation_rows": int(len(eval_df)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            )
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        for filename in ["predictions.csv", "inference_metrics.json"]:
            bucket.blob(
                f"{prefix}/{filename}"
            ).upload_from_filename(
                os.path.join(temp_dir, filename)
            )
            print(f"Uploaded: {prefix}/{filename}")

        print(f"Run ID: {args.run_id}")
        print(f"Inference accuracy: {metrics['accuracy']:.4f}")
        print("Inference completed successfully.")


if __name__ == "__main__":
    main()