from pathlib import Path
import pandas as pd
import fastf1

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FOLDER = BASE_DIR / "DATA" / "raw"
OUTPUT_FOLDER = BASE_DIR / "DATA" / "processed"

fastf1.Cache.enable_cache(str(BASE_DIR / "cache"))

metadata_rows = []

for season_folder in sorted(RAW_FOLDER.iterdir()):

    if not season_folder.is_dir():
        continue

    year = int(season_folder.name)

    print(f"\nLoading official schedule for {year}")

    schedule = fastf1.get_event_schedule(year)

    for _, event in schedule.iterrows():

        event_name = event["EventName"]

        if event_name == "Pre-Season Testing":
            continue

        race_folder = season_folder / event_name

        if not race_folder.exists():
            print(f"Missing folder : {event_name}")
            continue

        metadata_rows.append({

            "Season": year,

            "Round": int(event["RoundNumber"]),

            "Race": event_name,

            "Country": event["Country"],

            "Location": event["Location"],

            "EventDate": event["EventDate"]

        })

metadata = pd.DataFrame(metadata_rows)

metadata = metadata.sort_values(
    ["Season", "Round"]
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

metadata.to_csv(
    OUTPUT_FOLDER / "metadata.csv",
    index=False
)

print("\nMetadata Generated Successfully\n")

print(metadata.head())

print("\nRows :", len(metadata))