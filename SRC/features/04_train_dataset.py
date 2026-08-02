from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED = BASE_DIR / "DATA" / "processed"

df = pd.read_csv(
    PROCESSED / "feature_dataset.csv"
)

print("Sorting dataset...")

df = df.sort_values(
    ["Season", "Round"]
).reset_index(drop=True)

print("Creating rolling features...")

# -------------------------
# Driver Form
# -------------------------

df["DriverForm3"] = (
    df.groupby("Driver")["Finish"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(3, min_periods=1)
           .mean()
      )
)

df["DriverForm5"] = (
    df.groupby("Driver")["Finish"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(5, min_periods=1)
           .mean()
      )
)

df["DriverForm10"] = (
    df.groupby("Driver")["Finish"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(10, min_periods=1)
           .mean()
      )
)

# -------------------------
# Driver Points
# -------------------------

df["DriverPoints5"] = (
    df.groupby("Driver")["Points"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(5, min_periods=1)
           .mean()
      )
)

# -------------------------
# Grid Position
# -------------------------

df["DriverGrid5"] = (
    df.groupby("Driver")["Grid"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(5, min_periods=1)
           .mean()
      )
)

# -------------------------
# Team Form
# -------------------------

df["TeamForm5"] = (
    df.groupby("Team")["Finish"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(5, min_periods=1)
           .mean()
      )
)

# -------------------------
# Position Gain
# -------------------------

df["PositionsGained"] = (
    df["Grid"] -
    df["Finish"]
)

df["AvgGain5"] = (
    df.groupby("Driver")["PositionsGained"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(5, min_periods=1)
           .mean()
      )
)

# -------------------------
# DNF
# -------------------------

df["DNF"] = (
    df["Finish"].isna()
).astype(int)

df["DNFRate10"] = (
    df.groupby("Driver")["DNF"]
      .transform(
          lambda x:
          x.shift(1)
           .rolling(10, min_periods=1)
           .mean()
      )
)

# -------------------------
# Circuit History
# -------------------------

df["TrackHistory"] = (
    df.groupby(
        ["Driver", "Race"]
    )["Finish"]
    .transform(
        lambda x:
        x.shift(1)
         .expanding()
         .mean()
    )
)

# -------------------------
# Fill Missing Values
# -------------------------

df.fillna(0, inplace=True)

# -------------------------
# Save
# -------------------------

output = PROCESSED / "training_dataset.csv"

df.to_csv(
    output,
    index=False
)

print("\nTraining Dataset Created")

print()

print(df.head())

print()

print(df.shape)

print()

print(df.columns.tolist())