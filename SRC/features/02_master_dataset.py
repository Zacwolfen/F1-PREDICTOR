from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FOLDER = BASE_DIR / "DATA" / "raw"
PROCESSED_FOLDER = BASE_DIR / "DATA" / "processed"

metadata = pd.read_csv(
    PROCESSED_FOLDER / "metadata.csv"
)

master_dataset = []

successful = 0
failed = 0

print("\n========== BUILDING MASTER DATASET ==========\n")

for _, row in metadata.iterrows():

    season = int(row["Season"])
    round_number = int(row["Round"])
    race = row["Race"]

    race_folder = RAW_FOLDER / str(season) / race

    results_file = race_folder / "results.csv"

    if not results_file.exists():

        print(f"Missing results.csv -> {race}")
        failed += 1
        continue

    try:

        results = pd.read_csv(results_file)

        race_df = results[
            [
                "Abbreviation",
                "TeamName",
                "GridPosition",
                "Position",
                "Points"
            ]
        ].copy()

        race_df.rename(
            columns={
                "Abbreviation": "Driver",
                "TeamName": "Team",
                "GridPosition": "Grid",
                "Position": "Finish"
            },
            inplace=True
        )

        race_df["Season"] = season
        race_df["Round"] = round_number
        race_df["Race"] = race
        race_df["Country"] = row["Country"]
        race_df["Location"] = row["Location"]
        race_df["EventDate"] = row["EventDate"]

        race_df["Winner"] = (
            race_df["Finish"] == 1
        ).astype(int)

        master_dataset.append(race_df)

        successful += 1

        print(
            f"✓ Round {round_number:02d} : {race}"
        )

    except Exception as e:

        failed += 1

        print(
            f"✗ {race} -> {e}"
        )

if len(master_dataset) == 0:
    raise RuntimeError("No races were loaded.")

master_dataset = pd.concat(
    master_dataset,
    ignore_index=True
)

master_dataset = master_dataset.sort_values(
    ["Season", "Round", "Finish"]
).reset_index(drop=True)

master_dataset.to_csv(
    PROCESSED_FOLDER / "master_dataset.csv",
    index=False
)

print("\n==============================")
print("MASTER DATASET CREATED")
print("==============================")
print(f"Rows        : {len(master_dataset)}")
print(f"Columns     : {len(master_dataset.columns)}")
print(f"Successful  : {successful}")
print(f"Failed      : {failed}")

print("\nColumns:")
print(master_dataset.columns.tolist())

print("\nFirst Five Rows:\n")
print(master_dataset.head())
