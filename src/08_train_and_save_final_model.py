import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = os.path.join("data", "processed", "chelsea_features_odds.csv")
MODEL_PATH = os.path.join("outputs", "final_rf_model.joblib")

# Load data
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

label_map = {"Loss": 0, "Draw": 1, "Win": 2}
inv_label_map = {v: k for k, v in label_map.items()}
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

# Train final model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# Evaluate (optional but useful)
y_pred = model.predict(X_test)
print("Test size:", len(y_test))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Loss", "Draw", "Win"]))

# Save model bundle (model + features + label maps)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
bundle = {
    "model": model,
    "features": FEATURES,
    "label_map": label_map,
    "inv_label_map": inv_label_map,
}
joblib.dump(bundle, MODEL_PATH)

print("\nSaved model bundle to:", MODEL_PATH)