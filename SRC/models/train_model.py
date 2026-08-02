import os

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

import joblib

features = pd.read_csv(
    "../../data/processed/features.csv"
)

print(features.head())

print(features.shape)

driver_encoder = LabelEncoder()

team_encoder = LabelEncoder()

features["Driver"] = driver_encoder.fit_transform(
    features["Driver"]
)

features["Team"] = team_encoder.fit_transform(
    features["Team"]
)

X = features.drop(
    columns=["Winner"]
)

y = features["Winner"]

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy : {accuracy}")

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_

})

print(

    importance.sort_values(

        by="Importance",

        ascending=False

    )

)

os.makedirs(
    "../../models",
    exist_ok=True
)

import os

print("Current Working Directory:")
print(os.getcwd())

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/random_forest.pkl"
)

joblib.dump(
    driver_encoder,
    "models/driver_encoder.pkl"
)

joblib.dump(
    team_encoder,
    "models/team_encoder.pkl"
)

