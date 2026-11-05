"""Train the thesis baseline and export a model bundle plus measured reports."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS, build_match_features, build_team_strength
from .preprocessing import CLASSES, load_mapping, prepare_matches, prepare_players

MODEL_PARAMS = {
    "iterations": 700,
    "learning_rate": 0.04,
    "depth": 6,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "verbose": 100,
    # Packaging setting only: keep CatBoost's auxiliary files out of the repo.
    "allow_writing_files": False,
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--team-map", type=Path, required=True)
    parser.add_argument("--target-column", default="target_result")
    parser.add_argument("--result-map", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output must be a new or empty directory; choose another --output.")

    team_map = load_mapping(args.team_map)
    result_map = load_mapping(args.result_map)
    players = prepare_players(pd.read_csv(args.players), team_map)
    matches = prepare_matches(pd.read_csv(args.matches), args.target_column, result_map)
    strength = build_team_strength(players)
    X = build_match_features(matches, strength)
    y = matches["target_result"]
    counts = y.value_counts()
    if set(counts.index) != set(CLASSES) or counts.min() < 2:
        raise ValueError("Training requires all three outcome classes, each with at least two rows.")
    # Exact split specified in Appendix B. sklearn also validates split capacity.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    model = CatBoostClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    predictions = np.asarray(model.predict(X_test)).reshape(-1)
    probabilities = model.predict_proba(X_test)
    labels = [str(label) for label in model.classes_]
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS, "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output / "model.cbm"))
    strength.to_csv(args.output / "team_strength.csv", index=False)
    write_json(args.output / "team_map.json", team_map)
    write_json(args.output / "result_map.json", result_map)
    write_json(args.output / "metrics.json", {"accuracy": accuracy, "classification_report": report})
    (args.output / "classification_report.txt").write_text(
        classification_report(y_test, predictions, labels=labels, zero_division=0),
        encoding="utf-8",
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        args.output / "confusion_matrix.csv", index_label="true_label",
    )
    importance.to_csv(args.output / "feature_importance.csv", index=False)
    identity_columns = [name for name in ("match_id", "competition_name", "date_utc") if name in matches]
    held_out = matches.loc[X_test.index, identity_columns + ["home_team", "away_team", "target_result"]].copy()
    held_out["prediction"] = predictions
    for index, label in enumerate(labels):
        held_out[f"probability_{label}"] = probabilities[:, index]
    held_out.to_csv(args.output / "test_predictions.csv", index_label="source_row_index")
    write_json(args.output / "metadata.json", {
        "feature_columns": FEATURE_COLUMNS,
        "class_order": labels,
        "model_parameters": model.get_all_params(),
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "train_row_indices": X_train.index.tolist(),
        "test_row_indices": X_test.index.tolist(),
        "target_column": args.target_column,
        "python_version": platform.python_version(),
        "input_sha256": {
            "players": hashlib.sha256(args.players.read_bytes()).hexdigest(),
            "matches": hashlib.sha256(args.matches.read_bytes()).hexdigest(),
        },
        "versions": {name: version(name) for name in (
            "catboost", "scikit-learn", "pandas", "numpy", "matplotlib",
        )},
    })
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    ordered = importance.sort_values("importance")
    ax.barh(ordered["feature"], ordered["importance"])
    ax.set_xlabel("CatBoost feature importance")
    fig.tight_layout()
    fig.savefig(args.output / "feature_importance.png", dpi=160)
    plt.close(fig)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved model and reports to {args.output.resolve()}")


if __name__ == "__main__":
    main()
