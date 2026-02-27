from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import re

app = Flask(__name__)

MODEL_PATH = os.path.join("outputs", "final_rf_model.joblib")

def parse_scorelines(scorelines: str):
    parts = [p.strip() for p in scorelines.split(",") if p.strip()]
    if len(parts) != 5:
        raise ValueError("Enter exactly 5 scorelines, separated by commas.")

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

    goal_diff = gf - ga
    win_rate = wins / 5.0
    return float(points), float(gf), float(ga), float(goal_diff), float(win_rate)

def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run src/08_train_and_save_final_model.py first.")
    return joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probs = None
    confidence = None
    error = None
    used = None

    if request.method == "POST":
        try:
            bundle = load_bundle()
            model = bundle["model"]
            FEATURES = bundle["features"]
            inv_label_map = bundle["inv_label_map"]

            is_home = int(request.form.get("is_home"))
            odds_win = float(request.form.get("odds_win"))
            odds_draw = float(request.form.get("odds_draw"))
            odds_loss = float(request.form.get("odds_loss"))
            last5_scores = request.form.get("last5_scores", "").strip()

            formpoints_5, goalsfor_5, goalsagainst_5, goaldiff_5, winrate_5 = parse_scorelines(last5_scores)

            row = {
                "IsHome": is_home,
                "FormPoints_5": formpoints_5,
                "GoalsFor_5": goalsfor_5,
                "GoalsAgainst_5": goalsagainst_5,
                "GoalDiff_5": goaldiff_5,
                "WinRate_5": winrate_5,
                "Odds_Win": odds_win,
                "Odds_Draw": odds_draw,
                "Odds_Loss": odds_loss,
            }

            X = pd.DataFrame([row])[FEATURES]
            pred_class = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]

            labels = [inv_label_map[c] for c in model.classes_]
            probs = {labels[i]: round(float(proba[i]), 3) for i in range(len(labels))}
            result = inv_label_map[pred_class]
            used = row

            # Determine prediction confidence
            max_prob = max(probs.values())
            if max_prob >= 0.60:
                confidence = "High"
            elif max_prob >= 0.45:
                confidence = "Medium"
            else:
                confidence = "Low"

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        probs=probs,
        confidence=confidence,
        error=error,
        used=used
    )
if __name__ == "__main__":
    app.run(debug=True)