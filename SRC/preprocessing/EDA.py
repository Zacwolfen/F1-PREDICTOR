import pandas as pd
import matplotlib.pyplot as plt
results = pd.read_csv(
    "../../data/raw/2024/Bahrain Grand Prix/results.csv"
)
print(results.head())
print(results.info())
print(results.describe())
print(results.columns)
print(results.isnull().sum())
print(results["TeamName"].unique())
print(results["Status"].unique())
print(
    results["TeamName"].value_counts()
)
results["GridPosition"].hist(
    bins=20
)

plt.title("Grid Positions")

plt.xlabel("Grid")

plt.ylabel("Drivers")

plt.show()
laps = pd.read_csv(
    "../../data/raw/2024/Bahrain Grand Prix/laps.csv"
)
print(laps.head())

print(laps.info())
print(laps.columns)

weather = pd.read_csv(
    "../../data/raw/2024/Bahrain Grand Prix/weather.csv"
)

print(weather.head())

print(weather.info())

