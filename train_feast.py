from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    train_ids = feature_data[
        feature_data["sample_id"].str.startswith("train_")
    ][["sample_id", "event_timestamp"]]

    eval_ids = feature_data[
        feature_data["sample_id"].str.startswith("eval_")
    ][["sample_id", "event_timestamp"]]

    store = FeatureStore(repo_path=str(REPO))

    def retrieve(entity_df):
        return store.get_historical_features(
            entity_df=entity_df,
            features=[f"iris_features:{c}" for c in FEATURES],
        ).to_df()

    train_features = retrieve(train_ids).merge(
        labels, on="sample_id", validate="one_to_one"
    )
    eval_features = retrieve(eval_ids).merge(
        labels, on="sample_id", validate="one_to_one"
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(train_features[FEATURES], train_features["species"])
    predictions = model.predict(eval_features[FEATURES])
    accuracy = accuracy_score(eval_features["species"], predictions)

    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump(model, artifact_dir / "feast_model.joblib")

    print("Training rows:", len(train_features))
    print("Evaluation rows:", len(eval_features))
    print(f"Evaluation accuracy: {accuracy:.4f}")
    print("Saved:", artifact_dir / "feast_model.joblib")


if __name__ == "__main__":
    main()
