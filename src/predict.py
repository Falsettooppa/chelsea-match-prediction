import argparse
import joblib
import os
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("outputs", "final_rf_model.joblib")

def compute_form_features(last5: str):
    """
    last5 example: "W,W,D,L,W" (exactly 5 results)
    Returns: FormPoints_5, GoalsFor_5, GoalsAgainst_5, GoalDiff_5, WinRate_5
    NOTE: GoalsFor/Against cannot be derived from W/D/L alone, so we set them to 0 by default.
          If you know the last 5 scorelines, you can extend this later.
    """
    tokens = [t.strip().upper() for t in last5.split(",") if t.strip()]
    if len(tokens) != 5 or any(t not in {"W", "D", "L"} for t in tokens):
        raise ValueError("last5 must be exactly 5 items like: W,W,D,L,W")

    points = [3 if t == "W" else 1 if t == "D" else 0 for t in tokens]
    form_points = float(sum(points))
    win_rate = float(sum(1 for t in tokens if t == "W") / 5.0)

    # Goals features unknown from W/D/L only:
    goals_for = 0.0
    goals_against = 0.0
    goal_diff = goals_for - goals_against

    return form_points, goals_for, goals_against, goal_diff, win_rate

def main():
    parser = argparse.ArgumentParser(description="Chelsea match outcome predictor (RF + odds + form)")
    parser.add_argument("--is_home", type=int, choices=[0, 1], required=True, help="1 if Chelsea is home, else 0")
    parser.add_argument("--odds_win", type=float, required=True, help="Decimal odds for Chelsea Win")
    parser.add_argument("--odds_draw", type=float, required=True, help="Decimal odds for Draw")
    parser.add_argument("--odds_loss", type=float, required=True, help="Decimal odds for Chelsea Loss")

    # Option A: Provide form features directly (best/most accurate)
    parser.add_argument("--formpoints_5", type=float, help="Rolling points last 5 (0-15)")
    parser.add_argument("--goalsfor_5", type=float, help="Rolling goals scored last 5")
    parser.add_argument("--goalsagainst_5", type=float, help="Rolling goals conceded last 5")
    parser.add_argument("--goaldiff_5", type=float, help="Rolling goal diff last 5")
    parser.add_argument("--winrate_5", type=float, help="Rolling win rate last 5 (0-1)")

    # Option B: Quick fallback using last5 W/D/L only (goals default to 0)
    parser.add_argument("--last5", type=str, help='Last 5 results like "W,W,D,L,W" (goals default to 0)')

    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/08_train_and_save_final_model.py first.")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    FEATURES = bundle["features"]
    inv_label_map = bundle["inv_label_map"]

    # Build features
    if all(v is not None for v in [args.formpoints_5, args.goalsfor_5, args.goalsagainst_5, args.goaldiff_5, args.winrate_5]):
        formpoints_5 = args.formpoints_5
        goalsfor_5 = args.goalsfor_5
        goalsagainst_5 = args.goalsagainst_5
        goaldiff_5 = args.goaldiff_5
        winrate_5 = args.winrate_5
    elif args.last5:
        formpoints_5, goalsfor_5, goalsagainst_5, goaldiff_5, winrate_5 = compute_form_features(args.last5)
        print("Note: last5 mode used; GoalsFor_5 and GoalsAgainst_5 default to 0. For best accuracy, pass goals features.")
    else:
        raise ValueError("Provide either the 5 form feature args OR --last5.")

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

    # Model classes correspond to encoded labels 0/1/2
    # Map them to Loss/Draw/Win
    labels = [inv_label_map[c] for c in model.classes_]
    proba_dict = {labels[i]: float(proba[i]) for i in range(len(labels))}

    print("\nPrediction:", inv_label_map[pred_class])
    print("Probabilities:", {k: round(v, 3) for k, v in proba_dict.items()})

if __name__ == "__main__":
    main()