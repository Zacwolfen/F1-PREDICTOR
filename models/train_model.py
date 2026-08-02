from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET = (
    BASE_DIR /
    "DATA" /
    "processed" /
    "training_dataset.csv"
)

MODEL_FOLDER = BASE_DIR / "models"

MODEL_FOLDER.mkdir(
    exist_ok=True
)

print("Loading dataset...")

df = pd.read_csv(DATASET)

print(df.shape)

# ----------------------------
# Remove columns that should not
# be used for training
# ----------------------------

drop_columns = [

    # Targets
    "Finish",
    "Winner",

    # Text
    "Race",
    "Country",
    "Location",
    "EventDate",

    # POST-RACE FEATURES
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

    # Weather measured during race
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Rainfall"

]

X = df.drop(
    columns=drop_columns
)

print()

print("Features Used")

print("----------------------")

for column in X.columns:
    print(column)

print("----------------------")

y = df["Finish"]

# ----------------------------
# Encode text columns
# ----------------------------

X = pd.get_dummies(
    X,
    columns=[
        "Driver",
        "Team"
    ]
)

print()

print("Features :", X.shape[1])

# ----------------------------
# Train/Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    shuffle=False

)

print()

print("Training Rows :", len(X_train))

print("Testing Rows :", len(X_test))

# ----------------------------
# Train Model
# ----------------------------

print()

print("Training Random Forest...")

model = RandomForestRegressor(

    n_estimators=500,

    random_state=42,

    max_depth=15,

    n_jobs=-1

)

model.fit(

    X_train,

    y_train

)

# ----------------------------
# Predict
# ----------------------------

predictions = model.predict(

    X_test

)

# ----------------------------
# Metrics
# ----------------------------

mae = mean_absolute_error(

    y_test,

    predictions

)

mse = mean_squared_error(

    y_test,

    predictions

)

rmse = mse ** 0.5

r2 = r2_score(

    y_test,

    predictions

)

print()

print("==========================")

print("MODEL RESULTS")

print("==========================")

print(f"MAE  : {mae:.3f}")

print(f"RMSE : {rmse:.3f}")

print(f"R²   : {r2:.3f}")

print("==========================")

# ----------------------------
# Save Model
# ----------------------------

joblib.dump(

    model,

    MODEL_FOLDER / "random_forest.pkl"

)

print()

print("Model Saved")

print(

    MODEL_FOLDER /
    "random_forest.pkl"

)
