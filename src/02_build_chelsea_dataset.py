import pandas as pd
import os

IN_PATH = os.path.join("data", "processed", "epl_all_seasons.csv")
OUT_PATH = os.path.join("data", "processed", "chelsea_matches.csv")

df = pd.read_csv(IN_PATH)

# --- Basic sanity checks ---
required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# --- Filter to Chelsea matches only ---
chelsea = df[(df["HomeTeam"] == "Chelsea") | (df["AwayTeam"] == "Chelsea")].copy()

# --- Convert Date to datetime (football-data uses day-first dates) ---
chelsea["Date"] = pd.to_datetime(chelsea["Date"], dayfirst=True, errors="coerce")

# Drop invalid dates and sort chronologically
chelsea = chelsea.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# --- Create target label from Chelsea perspective: Win/Draw/Loss ---
def chelsea_outcome(row):
    # FTR: H=home win, D=draw, A=away win
    if row["HomeTeam"] == "Chelsea":
        return "Win" if row["FTR"] == "H" else "Draw" if row["FTR"] == "D" else "Loss"
    else:
        return "Win" if row["FTR"] == "A" else "Draw" if row["FTR"] == "D" else "Loss"

chelsea["Target"] = chelsea.apply(chelsea_outcome, axis=1)

# --- Helper columns ---
chelsea["IsHome"] = (chelsea["HomeTeam"] == "Chelsea").astype(int)
chelsea["Opponent"] = chelsea.apply(lambda r: r["AwayTeam"] if r["IsHome"] == 1 else r["HomeTeam"], axis=1)
chelsea["ChelseaGoals"] = chelsea.apply(lambda r: r["FTHG"] if r["IsHome"] == 1 else r["FTAG"], axis=1)
chelsea["OppGoals"] = chelsea.apply(lambda r: r["FTAG"] if r["IsHome"] == 1 else r["FTHG"], axis=1)

# Keep clean core columns (we add engineered features in the next script)
keep_cols = [
    "Date", "SeasonTag", "HomeTeam", "AwayTeam", "Opponent",
    "IsHome", "FTHG", "FTAG", "ChelseaGoals", "OppGoals", "FTR", "Target"
]
keep_cols = [c for c in keep_cols if c in chelsea.columns]
chelsea_clean = chelsea[keep_cols].copy()

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
chelsea_clean.to_csv(OUT_PATH, index=False)

print("Chelsea dataset saved:", OUT_PATH)
print("Rows:", chelsea_clean.shape[0])
print("Target distribution:")
print(chelsea_clean["Target"].value_counts())
print("\nSample rows:")
print(chelsea_clean.head(5))