import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_PATH = os.path.join("data", "processed", "chelsea_features_odds.csv")
df = pd.read_csv(DATA_PATH)

# Ensure Date is datetime and keep chronological order
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Encode target
label_map = {"Loss": 0, "Draw": 1, "Win": 2}
df["TargetEncoded"] = df["Target"].map(label_map)

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
y = df["TargetEncoded"]

# Time-aware split (train first 80%, test last 20%)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Test size:", len(y_test))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Loss", "Draw", "Win"]
))

# Feature importance (for Chapter 4)
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop Feature Importances:")
print(importances.head(10))