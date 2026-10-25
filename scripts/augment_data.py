import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/train.csv")
ADDED_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

if ADDED_ROWS <= 0:
    raise ValueError("Number of added rows must be positive.")

df = pd.read_csv(DATA_PATH)
features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

# Deterministically create new, slightly perturbed training samples.
new_rows = []
for i in range(ADDED_ROWS):
    source = df.iloc[i % len(df)].copy()
    offset = ((i % 5) - 2) * 0.01

    for feature in features:
        source[feature] = max(0.0, float(source[feature]) + offset)

    new_rows.append(source)

augmented = pd.concat(
    [df, pd.DataFrame(new_rows)],
    ignore_index=True,
)
augmented.to_csv(DATA_PATH, index=False)

print(f"Previous training rows: {len(df)}")
print(f"Added rows: {ADDED_ROWS}")
print(f"New training rows: {len(augmented)}")
print("Evaluation dataset unchanged.")
