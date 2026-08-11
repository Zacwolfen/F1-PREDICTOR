from pathlib import Path

import pandas as pd

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET = (
    BASE_DIR
    / "DATA"
    / "processed"
    / "training_dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model.pkl"
)

print("Loading trained model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")

print()

print("Loading feature dataset...")

df = pd.read_csv(DATASET)

print(df.shape)

DROP_COLUMNS = [

    "Finish",
    "Winner",

    "Race",
    "Country",
    "Location",
    "EventDate",

    "AverageLap",
    "FastestLap",
    "MedianLap",

    "LapConsistency",

    "CompletedLaps",

    "AverageSector1",
    "AverageSector2",
    "AverageSector3",

    "AverageTyreLife",

    "NumberOfStints",

    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Rainfall"

]

X = df.drop(
    columns=DROP_COLUMNS
)

X = pd.get_dummies(

    X,

    columns=[
        "Driver",
        "Team"
    ]

)

model_features = model.feature_names_in_

X = X.reindex(
    columns=model_features,
    fill_value=0
)

predictions = model.predict(X)

df["PredictedFinish"] = predictions

prediction_table = df[
    [
        "Driver",
        "Team",
        "PredictedFinish"
    ]
].copy()

prediction_table = prediction_table.sort_values(
    by="PredictedFinish"
)

print()

print("=======================================")
print("PREDICTED FINISHING ORDER")
print("=======================================")

print(prediction_table.head(20))

