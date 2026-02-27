import argparse
import joblib
import os
import re
import pandas as pd

MODEL_PATH = os.path.join("outputs", "final_rf_model.joblib")

def parse_scorelines(scorelines: str):
    """
    scorelines example:
      "2-1, 0-0, 1-2, 3-0, 1-1"
    Interpreted from Chelsea perspective (ChelseaGoals-OppGoals) for each match.
    Returns rolling features for last 5:
      FormPoints_5, GoalsFor_5, GoalsAgainst_5, GoalDiff_5, WinRate_5
    """
    parts = [p.strip() for p in scorelines.split(",") if p.strip()]
    if len(parts) != 5:
        raise ValueError("Provide exactly 5 scorelines separated by commas, e.g. '2-1,0-0,1-2,3-0,1-1'.")

    gf = ga = wins = points = 0
    for s in parts:
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
        if not m:
            raise ValueError(f"Invalid scoreline '{s}'. Use format like 2-1.")
        chelsea_g = int(m.group(1))
        opp_g = int(m.group(2))

        gf += chelsea_g
        ga += opp_g

        if chelsea_g > opp_g:
            points += 3
            wins += 1
        elif chelsea_g == opp_g:
            points += 1
        else:
            points += 0

    goal_diff = gf - ga
    win_rate = wins / 5.0
    return float(points), float(gf), float(ga), float(goal_diff), float(win_rate)

def main():
    parser = argparse.ArgumentParser(description="Chelsea predictor (RF + rolling form + odds)")

    parser.add_argument("--is_home", type=int, choices=[0, 1], required=True, help="1 if Chelsea is home, else 0")
    parser.add_argument("--odds_win", type=float, required=True, help="Decimal odds for Chelsea Win")
    parser.add_argument("--odds_draw", type=float, required=True, help="Decimal odds for Draw")
    parser.add_argument("--odds_loss", type=float, required=True, help="Decimal odds for Chelsea Loss")

    # Option A: supply rolling stats directly
    parser.add_argument("--formpoints_5", type=float)
    parser.add_argument("--goalsfor_5", type=float)
    parser.add_argument("--goalsagainst_5", type=float)
    parser.add_argument("--goaldiff_5", type=float)
    parser.add_argument("--winrate_5", type=float)

    # Option B: supply last 5 scorelines (best for demo)
    parser.add_argument("--last5_scores", type=str, help='Example: "2-1,0-0,1-2,3-0,1-1" (Chelsea perspective)')

    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/08_train_and_save_final_model.py first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    FEATURES = bundle["features"]
    inv_label_map = bundle["inv_label_map"]

    # Build form features
    if args.last5_scores:
        formpoints_5, goalsfor_5, goalsagainst_5, goaldiff_5, winrate_5 = parse_scorelines(args.last5_scores)
    elif all(v is not None for v in [args.formpoints_5, args.goalsfor_5, args.goalsagainst_5, args.goaldiff_5, args.winrate_5]):
        formpoints_5 = args.formpoints_5
        goalsfor_5 = args.goalsfor_5
        goalsagainst_5 = args.goalsagainst_5
        goaldiff_5 = args.goaldiff_5
        winrate_5 = args.winrate_5
    else:
        raise ValueError("Provide either --last5_scores OR all 5 rolling feature arguments.")

    row = {
        "IsHome": args.is_home,
        "FormPoints_5": formpoints_5,
        "GoalsFor_5": goalsfor_5,
        "GoalsAgainst_5": goalsagainst_5,
        "GoalDiff_5": goaldiff_5,
        "WinRate_5": winrate_5,
        "Odds_Win": args.odds_win,
        "Odds_Draw": args.odds_draw,
        "Odds_Loss": args.odds_loss,
    }

    X = pd.DataFrame([row])[FEATURES]

    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    labels = [inv_label_map[c] for c in model.classes_]
    proba_dict = {labels[i]: float(proba[i]) for i in range(len(labels))}

    print("\nINPUTS USED:")
    for k, v in row.items():
        print(f"  {k}: {v}")

    print("\nPREDICTION:", inv_label_map[pred_class])
    print("PROBABILITIES:", {k: round(v, 3) for k, v in proba_dict.items()})

if __name__ == "__main__":
    main()