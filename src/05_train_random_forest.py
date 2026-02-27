import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
DATA_PATH = os.path.join("data", "processed", "chelsea_features.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

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
]

X = df[FEATURES]
y = df["TargetEncoded"]

# Time-aware split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train Random Forest (handles non-linearity + imbalance better)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Loss", "Draw", "Win"]
))