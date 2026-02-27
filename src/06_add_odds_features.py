import pandas as pd
import os

# Paths
EPL_PATH = os.path.join("data", "processed", "epl_all_seasons.csv")
FEATURES_PATH = os.path.join("data", "processed", "chelsea_features.csv")
OUT_PATH = os.path.join("data", "processed", "chelsea_features_odds.csv")

# Load datasets
epl = pd.read_csv(EPL_PATH)
epl["Date"] = pd.to_datetime(epl["Date"], dayfirst=True, errors="coerce")

features = pd.read_csv(FEATURES_PATH)
features["Date"] = pd.to_datetime(features["Date"], errors="coerce")

# Keep only Chelsea matches from EPL (with odds still present)
epl_chelsea = epl[
    (epl["HomeTeam"] == "Chelsea") | (epl["AwayTeam"] == "Chelsea")
].copy()

# Select odds columns
odds_cols = ["B365H", "B365D", "B365A"]
available_odds = [c for c in odds_cols if c in epl_chelsea.columns]

if not available_odds:
    raise ValueError("Betting odds columns not found in EPL dataset.")

odds = epl_chelsea[["Date", "HomeTeam", "AwayTeam"] + available_odds]

# Merge odds into engineered feature dataset
df = features.merge(
    odds,
    on=["Date", "HomeTeam", "AwayTeam"],
    how="left"
)

# Normalize odds relative to Chelsea perspective
df["Odds_Win"] = df.apply(
    lambda r: r["B365H"] if r["IsHome"] == 1 else r["B365A"],
    axis=1
)
df["Odds_Draw"] = df["B365D"]
df["Odds_Loss"] = df.apply(
    lambda r: r["B365A"] if r["IsHome"] == 1 else r["B365H"],
    axis=1
)

# Save
df.to_csv(OUT_PATH, index=False)

print("Saved dataset with odds:", OUT_PATH)
print("Sample odds features:")
print(df[["Odds_Win", "Odds_Draw", "Odds_Loss"]].head())