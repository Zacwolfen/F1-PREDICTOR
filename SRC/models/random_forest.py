import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
df = pd.read_csv(
    "DATA/processed/training_dataset.csv"
)
print(df.head())

print(df.shape)

target = "Finish"

columns_to_drop = [

    "Finish",

    "Winner",

    "Points",

    "Driver",

    "Team",

    "Race",

    "Country",

    "Location",

    "EventDate"

]

X = df.drop(columns=columns_to_drop)

y = df[target]

print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

print(X_train.shape)

print(X_test.shape)

