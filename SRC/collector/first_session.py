from pathlib import Path
import fastf1

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"

CACHE.mkdir(exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE))
session = fastf1.get_session(
    2024,
    "Monaco Grand Prix",
    "R"
)
session.load()
print(session.event)
print(type(session))
print(session.results.head())
print(session.laps.head())
print(session.weather_data.head())
print(session.results.columns)

print(session.laps.columns)

print(session.weather_data.columns)

session.results.to_csv("data/monaco_2024_results.csv")