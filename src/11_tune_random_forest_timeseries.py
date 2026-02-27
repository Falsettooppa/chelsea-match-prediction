import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

DATA_PATH = os.path.join("data", "processed", "chelsea_features_odds.csv")
df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

label_map = {"Loss": 0, "Draw": 1, "Win": 2}
df["y"] = df["Target"].map(label_map)

FEATURES = [
    "IsHome",
    "FormPoints_5",
    "GoalsFor_5",
    "GoalsAgainst_5",
    "GoalDiff_5",
    "WinRate_5",
    "Odds_Win",
    "Odds_Draw",
    "Odds_Loss",
]

X = df[FEATURES]
y = df["y"]

# Time series split
tscv = TimeSeriesSplit(n_splits=5)

param_dist = {
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [6, 8, 10, 12, None],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"],
}

rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=25,
    cv=tscv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

search.fit(X, y)

print("Best Parameters:")
print(search.best_params_)
print("Best Cross-Validated Accuracy:", round(search.best_score_, 4))