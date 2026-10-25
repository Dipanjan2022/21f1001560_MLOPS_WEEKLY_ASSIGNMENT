import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
eval_df = pd.read_csv(DATA_DIR / "eval.csv")

target = "species"
features = [column for column in train_df.columns if column != target]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

model.fit(train_df[features], train_df[target])

predictions = model.predict(eval_df[features])
accuracy = accuracy_score(eval_df[target], predictions)

joblib.dump(model, ARTIFACT_DIR / "model.joblib")

metrics = {
    "accuracy": float(accuracy),
    "training_rows": len(train_df),
    "evaluation_rows": len(eval_df),
    "model_type": "StandardScaler + LogisticRegression",
    "classification_report": classification_report(
        eval_df[target], predictions, output_dict=True
    ),
}

with open(ARTIFACT_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=2)

print(f"Training rows: {len(train_df)}")
print(f"Evaluation rows: {len(eval_df)}")
print(f"Evaluation accuracy: {accuracy:.4f}")
print("Saved artifacts/model.joblib and artifacts/metrics.json")
