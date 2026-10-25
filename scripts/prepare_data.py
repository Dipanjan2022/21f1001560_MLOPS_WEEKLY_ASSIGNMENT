from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
    "target": "species_id",
})

df["species"] = df["species_id"].map(
    dict(enumerate(iris.target_names))
)
df = df.drop(columns=["species_id"])

train_df, eval_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["species"],
)

train_df.to_csv(DATA_DIR / "train.csv", index=False)
eval_df.to_csv(DATA_DIR / "eval.csv", index=False)

print(f"Training rows: {len(train_df)}")
print(f"Evaluation rows: {len(eval_df)}")
print(f"Training class counts:\n{train_df['species'].value_counts()}")
print("Created data/train.csv and data/eval.csv")
