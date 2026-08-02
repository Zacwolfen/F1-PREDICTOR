from pathlib import Path
import os
import fastf1

ROOT = Path(__file__).resolve().parents[2]

CACHE = ROOT / "cache"
DATA = ROOT / "data" / "raw" / "2024"

CACHE.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE))

YEAR = 2024

schedule = fastf1.get_event_schedule(YEAR)

for _, race in schedule.iterrows():

    session = fastf1.get_session(
        YEAR,
        race["EventName"],
        "R"
    )

    session.load()

    folder = DATA / race["EventName"]
    folder.mkdir(parents=True, exist_ok=True)

    session.results.to_csv(
        f"{folder}/results.csv",
        index=False
    )

    session.laps.to_csv(
        f"{folder}/laps.csv",
        index=False
    )

    session.weather_data.to_csv(
        f"{folder}/weather.csv",
        index=False
    )

    print(f"Downloaded {race['EventName']}")


session.results.to_csv(folder / "results.csv", index=False)
session.laps.to_csv(folder / "laps.csv", index=False)
session.weather_data.to_csv(folder / "weather.csv", index=False)