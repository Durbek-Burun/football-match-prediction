"""Predict a match using the saved model and its training-time team statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .features import FEATURE_COLUMNS, build_match_features
from .preprocessing import CLASSES


def predict_match(home_team, away_team, team_strength, model) -> dict:
    matches = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
    X = build_match_features(matches, team_strength)
    if list(model.feature_names_) != FEATURE_COLUMNS:
        raise ValueError("Model feature names/order do not match this implementation.")
    labels = [str(label) for label in model.classes_]
    if set(labels) != set(CLASSES):
        raise ValueError("Model must contain the three expected match outcome classes.")
    prediction = str(np.asarray(model.predict(X)).reshape(-1)[0])
    probabilities = model.predict_proba(X)[0]
    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "probabilities": {label: float(p) for label, p in zip(labels, probabilities)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--home", required=True, help="Exact match-dataset team name")
    parser.add_argument("--away", required=True, help="Exact match-dataset team name")
    args = parser.parse_args()
    model = CatBoostClassifier()
    model.load_model(str(args.bundle / "model.cbm"))
    strength = pd.read_csv(args.bundle / "team_strength.csv")
    result = predict_match(args.home, args.away, strength, model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
