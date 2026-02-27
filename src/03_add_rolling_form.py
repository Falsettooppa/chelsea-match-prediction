import pandas as pd
import os

IN_PATH = os.path.join("data", "processed", "chelsea_matches.csv")
OUT_PATH = os.path.join("data", "processed", "chelsea_features.csv")

df = pd.read_csv(IN_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Map target to points
points_map = {"Win": 3, "Draw": 1, "Loss": 0}
df["Points"] = df["Target"].map(points_map)

WINDOW = 5  # last 5 matches

# Rolling form features (shifted to avoid data leakage)
df["FormPoints_5"] = df["Points"].shift(1).rolling(WINDOW).sum()
df["GoalsFor_5"] = df["ChelseaGoals"].shift(1).rolling(WINDOW).sum()
df["GoalsAgainst_5"] = df["OppGoals"].shift(1).rolling(WINDOW).sum()
df["GoalDiff_5"] = df["GoalsFor_5"] - df["GoalsAgainst_5"]
df["WinRate_5"] = df["Points"].shift(1).rolling(WINDOW).apply(
    lambda x: (x == 3).sum() / WINDOW
)

# Remove rows without enough history
df_features = df.dropna().reset_index(drop=True)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df_features.to_csv(OUT_PATH, index=False)

print("Feature dataset saved:", OUT_PATH)
print("Rows after rolling features:", df_features.shape[0])
print(df_features[
    ["Date", "Target", "FormPoints_5", "GoalsFor_5", "GoalsAgainst_5", "GoalDiff_5", "WinRate_5"]
].head(5))