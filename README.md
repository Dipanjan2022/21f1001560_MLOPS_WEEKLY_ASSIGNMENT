# IITM MLOps Week 1 Assignment

## Objective
Build a machine learning training and inference workflow using the Iris dataset, Google Cloud Storage (GCS), and Vertex AI Workbench.

## Dataset
- `train.csv`: training data
- `eval.csv`: evaluation data
- Target column: `species`
- Input features: sepal length, sepal width, petal length, and petal width

## Training workflow
The `train.py` script downloads the datasets from GCS, standardizes the features using `StandardScaler`, trains a `LogisticRegression` classifier, evaluates its performance, and uploads the trained model and metadata.

## Inference workflow
The `infer.py` script downloads the saved model, generates predictions on the evaluation dataset, calculates evaluation metrics, and uploads predictions and inference metrics to GCS.

## Experiment tracking
Two runs were executed with separate run IDs:
- `20261010T143000Z`
- `20261010T150000Z`

Each run stores its model, training metrics, run metadata, predictions, and inference metrics in its own GCS artifact folder.

## Technologies
- Python
- pandas
- scikit-learn
- joblib
- Google Cloud Storage
- Vertex AI Workbench