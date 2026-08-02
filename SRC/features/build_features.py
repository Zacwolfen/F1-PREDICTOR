import os
import pandas as pd

results = pd.read_csv(
    "../../data/raw/2024/Bahrain Grand Prix/results.csv"
)

features = results[
    [
        "Abbreviation",
        "TeamName",
        "GridPosition",
        "Position",
        "Points"
    ]
].copy()

features.rename(
    columns={
        "Abbreviation":"Driver",
        "TeamName":"Team",
        "GridPosition":"Grid",
        "Position":"Finish"
    },
    inplace=True
)

features["Grid"]


laps = pd.read_csv(
    "../../data/raw/2024/Bahrain Grand Prix/laps.csv"
)


laps["LapTime"] = pd.to_timedelta(
    laps["LapTime"],
    errors="coerce"
)

laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()


average_laps = (
    laps
    .groupby("Driver")["LapTimeSeconds"]
    .mean()
)

fastest_lap = (
    laps
    .groupby("Driver")["LapTimeSeconds"]
    .min()
)

lap_count = (
    laps
    .groupby("Driver")
    .size()
)

features["AverageLap"] = (
    features["Driver"]
    .map(average_laps)
)

features["FastestLap"] = (
    features["Driver"]
    .map(fastest_lap)
)

features["CompletedLaps"] = (
    features["Driver"]
    .map(lap_count)
)

features["PositionsGained"] = (
    features["Grid"]
    - features["Finish"]
)

features["Winner"] = (
    features["Finish"] == 1
).astype(int)

print(features.isnull().sum())

features.fillna(0, inplace=True)

os.makedirs(
    "../../data/processed",
    exist_ok=True
)

output_path = os.path.abspath("../../data/processed/features.csv")

print("Saving to:", output_path)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "features.csv"

print("Saving to:", OUTPUT_FILE)

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("File exists:", os.path.exists(output_path))

print(features.head())

print("\nFeature dataset created successfully!")

print(f"Rows: {len(features)}")
print(f"Columns: {len(features.columns)}")


