import os
import numpy as np
import pandas as pd

IN_PATH = os.path.join("data", "processed", "chelsea_features_odds.csv")
OUT_PATH = os.path.join("data", "processed", "chelsea_features_odds_plus.csv")

df = pd.read_csv(IN_PATH)

# Basic safety: avoid divide-by-zero
for c in ["Odds_Win", "Odds_Draw", "Odds_Loss"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Odds_Win", "Odds_Draw", "Odds_Loss"]).copy()

p_win_raw = 1.0 / df["Odds_Win"]
p_draw_raw = 1.0 / df["Odds_Draw"]
p_loss_raw = 1.0 / df["Odds_Loss"]

p_sum = p_win_raw + p_draw_raw + p_loss_raw

df["ImpP_Win"] = p_win_raw / p_sum
df["ImpP_Draw"] = p_draw_raw / p_sum
df["ImpP_Loss"] = p_loss_raw / p_sum
df["Overround"] = p_sum  # bookmaker margin proxy

df["LogOdds_Win"] = np.log(df["Odds_Win"])
df["LogOdds_Draw"] = np.log(df["Odds_Draw"])
df["LogOdds_Loss"] = np.log(df["Odds_Loss"])

df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
print(df[["Odds_Win","Odds_Draw","Odds_Loss","ImpP_Win","ImpP_Draw","ImpP_Loss","Overround"]].head())