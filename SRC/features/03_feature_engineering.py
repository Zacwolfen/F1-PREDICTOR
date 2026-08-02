from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FOLDER = BASE_DIR / "DATA" / "raw"
PROCESSED_FOLDER = BASE_DIR / "DATA" / "processed"

master = pd.read_csv(
    PROCESSED_FOLDER / "master_dataset.csv"
)

feature_rows = []

print("\n========== BUILDING FEATURE DATASET ==========\n")

for _, row in master.iterrows():

    season = int(row["Season"])
    race = row["Race"]
    driver = row["Driver"]

    race_folder = RAW_FOLDER / str(season) / race

    laps_file = race_folder / "laps.csv"
    weather_file = race_folder / "weather.csv"

    try:

        laps = pd.read_csv(laps_file)

        driver_laps = laps[
            laps["Driver"] == driver
        ].copy()

        if len(driver_laps) == 0:

            feature = row.to_dict()

            feature["AverageLap"] = np.nan
            feature["FastestLap"] = np.nan
            feature["MedianLap"] = np.nan
            feature["LapConsistency"] = np.nan
            feature["CompletedLaps"] = 0
            feature["AverageSector1"] = np.nan
            feature["AverageSector2"] = np.nan
            feature["AverageSector3"] = np.nan
            feature["AverageTyreLife"] = np.nan
            feature["NumberOfStints"] = 0

            weather = pd.read_csv(weather_file)

            feature["AirTemp"] = weather["AirTemp"].mean()
            feature["TrackTemp"] = weather["TrackTemp"].mean()
            feature["Humidity"] = weather["Humidity"].mean()
            feature["Pressure"] = weather["Pressure"].mean()
            feature["WindSpeed"] = weather["WindSpeed"].mean()

            if "Rainfall" in weather.columns:
                feature["Rainfall"] = weather["Rainfall"].max()
            else:
                feature["Rainfall"] = False

            feature_rows.append(feature)

            continue

        driver_laps["LapTime"] = pd.to_timedelta(
            driver_laps["LapTime"]
        ).dt.total_seconds()

        driver_laps["Sector1Time"] = pd.to_timedelta(
            driver_laps["Sector1Time"]
        ).dt.total_seconds()

        driver_laps["Sector2Time"] = pd.to_timedelta(
            driver_laps["Sector2Time"]
        ).dt.total_seconds()

        driver_laps["Sector3Time"] = pd.to_timedelta(
            driver_laps["Sector3Time"]
        ).dt.total_seconds()

        weather = pd.read_csv(weather_file)

        feature = row.to_dict()

        feature["AverageLap"] = driver_laps["LapTime"].mean()

        feature["FastestLap"] = driver_laps["LapTime"].min()

        feature["MedianLap"] = driver_laps["LapTime"].median()

        feature["LapConsistency"] = driver_laps["LapTime"].std()

        feature["CompletedLaps"] = len(driver_laps)

        feature["AverageSector1"] = driver_laps["Sector1Time"].mean()

        feature["AverageSector2"] = driver_laps["Sector2Time"].mean()

        feature["AverageSector3"] = driver_laps["Sector3Time"].mean()

        if "TyreLife" in driver_laps.columns:
            feature["AverageTyreLife"] = driver_laps["TyreLife"].mean()
        else:
            feature["AverageTyreLife"] = np.nan

        if "Stint" in driver_laps.columns:
            feature["NumberOfStints"] = driver_laps["Stint"].nunique()
        else:
            feature["NumberOfStints"] = np.nan

        feature["AirTemp"] = weather["AirTemp"].mean()

        feature["TrackTemp"] = weather["TrackTemp"].mean()

        feature["Humidity"] = weather["Humidity"].mean()

        feature["Pressure"] = weather["Pressure"].mean()

        feature["WindSpeed"] = weather["WindSpeed"].mean()

        if "Rainfall" in weather.columns:
            feature["Rainfall"] = weather["Rainfall"].max()
        else:
            feature["Rainfall"] = 0

        feature_rows.append(feature)

        print(
            f"✓ {season} {race} {driver}"
        )

    except Exception as e:

        print(
            f"✗ {season} {race} {driver} -> {e}"
        )

feature_dataset = pd.DataFrame(feature_rows)

feature_dataset.to_csv(
    PROCESSED_FOLDER / "feature_dataset.csv",
    index=False
)

print("\n==============================")
print("FEATURE DATASET CREATED")
print("==============================")

print(feature_dataset.head())

print()

print(feature_dataset.shape)