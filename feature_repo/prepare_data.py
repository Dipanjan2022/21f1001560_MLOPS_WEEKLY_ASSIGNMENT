from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET = "species"


def main():
    frames = []

    for split in ["train", "eval"]:
        df = pd.read_csv(PROJECT_ROOT / "data" / f"{split}.csv")

        required = FEATURES + [TARGET]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {split}.csv: {missing}")

        df = df[required].copy()
        df["sample_id"] = [f"{split}_{i:03d}" for i in range(len(df))]
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    # Synthetic, deterministic UTC timestamps for this static dataset.
    all_data["event_timestamp"] = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=len(all_data),
        freq="s",
        tz="UTC",
    )

    feature_data = all_data[
        ["sample_id", "event_timestamp"] + FEATURES
    ].copy()

    # Keep labels separate from the feature store.
    labels = all_data[["sample_id", TARGET]].copy()

    for col in FEATURES:
        feature_data[col] = feature_data[col].astype("float32")

    feature_path = ROOT / "data" / "iris_features.parquet"
    labels_path = ROOT / "data" / "iris_labels.csv"

    feature_data.to_parquet(feature_path, index=False)
    labels.to_csv(labels_path, index=False)

    print(f"Feature rows: {len(feature_data)}")
    print(f"Feature columns: {feature_data.columns.tolist()}")
    print(f"Labels saved: {labels_path}")
    print(f"Features saved: {feature_path}")
    print(feature_data.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
