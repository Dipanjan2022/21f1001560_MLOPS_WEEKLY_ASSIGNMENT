from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "feature_repo"
FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def main():
    feature_data = pd.read_parquet(
        REPO / "data" / "iris_features.parquet"
    )
    labels = pd.read_csv(REPO / "data" / "iris_labels.csv")

    sample_ids = ["eval_000", "eval_001"]
    entity_rows = [{"sample_id": sid} for sid in sample_ids]

    store = FeatureStore(repo_path=str(REPO))
    result = store.get_online_features(
        features=[f"iris_features:{c}" for c in FEATURES],
        entity_rows=entity_rows,
    ).to_dict()

    online_df = pd.DataFrame(result)
    online_df = online_df[FEATURES]

    model = joblib.load(ROOT / "artifacts" / "feast_model.joblib")
    predictions = model.predict(online_df)

    # Validate against predictions using the corresponding raw feature values.
    raw = feature_data.set_index("sample_id").loc[sample_ids, FEATURES]
    raw_predictions = model.predict(raw)

    print("Sample IDs:", sample_ids)
    print("Online features:")
    print(online_df.to_string(index=False))
    print("Feast online predictions:", predictions.tolist())
    print("Raw-feature predictions:", raw_predictions.tolist())
    print(
        "Predictions consistent:",
        (predictions == raw_predictions).all(),
    )

    print("\nActual species:")
    print(
        labels[labels["sample_id"].isin(sample_ids)]
        .set_index("sample_id")
        .loc[sample_ids, "species"]
        .to_dict()
    )


if __name__ == "__main__":
    main()
