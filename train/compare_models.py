from pathlib import Path

import pandas as pd

import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET = (
    BASE_DIR
    / "DATA"
    / "processed"
    / "training_dataset.csv"
)

MODEL_FOLDER = BASE_DIR / "models"

MODEL_FOLDER.mkdir(
    exist_ok=True
)

print("Loading dataset...")

df = pd.read_csv(DATASET)

print(df.shape)

print(df.head())

TARGET = "Finish"

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

y = df[TARGET]

X = pd.get_dummies(

    X,

    columns=[
        "Driver",
        "Team"
    ]

)

print()

print("Training Features")

print(X.columns.tolist())

print()

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    shuffle=False

)

print("\nTraining Rows :", len(X_train))
print("Testing Rows  :", len(X_test))

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        random_state=42
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=500,
        random_state=42
    ),

    "CatBoost": CatBoostRegressor(
        iterations=500,
        random_state=42,
        verbose=False
    )

}

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

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

    results.append({

        "Model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "ModelObject": model

    })

    print("Finished.")

    results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE"
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    results_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2"
        ]
    ]
)

best_model = results_df.iloc[0]

print("\n==============================")

print("BEST MODEL")

print("==============================")

print(best_model["Model"])

print("==============================")

joblib.dump(

    best_model["ModelObject"],

    MODEL_FOLDER / "best_model.pkl"

)

print("\nSaved:")

print(MODEL_FOLDER / "best_model.pkl")


