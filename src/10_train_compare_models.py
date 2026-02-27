import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

DATA_PATH = os.path.join("data", "processed", "chelsea_features_odds_plus.csv")
df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

label_map = {"Loss": 0, "Draw": 1, "Win": 2}
inv = {v: k for k, v in label_map.items()}
df["y"] = df["Target"].map(label_map)

FEATURES = [
    "IsHome",
    "FormPoints_5", "GoalsFor_5", "GoalsAgainst_5", "GoalDiff_5", "WinRate_5",
    "Odds_Win", "Odds_Draw", "Odds_Loss",
    "ImpP_Win", "ImpP_Draw", "ImpP_Loss", "Overround",
    "LogOdds_Win", "LogOdds_Draw", "LogOdds_Loss",
]

X = df[FEATURES]
y = df["y"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=800, max_depth=12, random_state=42, class_weight="balanced"
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=1200, max_depth=14, random_state=42, class_weight="balanced"
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=600, random_state=42
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print("\n====", name, "====")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, pred, target_names=["Loss", "Draw", "Win"]))