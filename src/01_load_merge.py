import pandas as pd
import glob
import os

RAW_PATTERN = os.path.join("data", "raw", "E0*.csv")  # matches E0_*.csv and similar

files = sorted(glob.glob(RAW_PATTERN))
if not files:
    raise FileNotFoundError(
        f"No EPL CSV files found in data/raw/. Expected pattern: {RAW_PATTERN}\n"
        "Make sure you downloaded Premier League files from football-data.co.uk and moved them into data/raw/."
    )

df_list = []
for fp in files:
    season_tag = os.path.splitext(os.path.basename(fp))[0]  # e.g., E0_1819
    df = pd.read_csv(fp)
    df["SeasonTag"] = season_tag
    df_list.append(df)

epl = pd.concat(df_list, ignore_index=True)

print("Loaded files:", len(files))
print("Combined dataset shape:", epl.shape)
print("Some columns:", list(epl.columns)[:25])
print(epl.head(3))

# Save combined dataset
out_path = os.path.join("data", "processed", "epl_all_seasons.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
epl.to_csv(out_path, index=False)
print("Saved:", out_path)