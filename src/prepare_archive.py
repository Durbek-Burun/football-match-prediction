"""Convert the supplied project tables into verified inputs for the thesis model.

Point --source to the extracted original project's data directory. The original
training code drops incomplete match rows; this adapter records those exclusions
and checks the saved aggregates against means rebuilt from the supplied players.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import TEAM_FEATURES, build_match_features, build_team_strength
from .preprocessing import PLAYER_METRICS, prepare_matches, prepare_players, require_columns

SOURCE_NAMES = (
    "players_with_team_stats_2024_2025.csv",
    "match_player_base_dataset_2024_2025.csv",
    "team_map_2024_2025.csv",
    "team_strength_base_2024_2025.csv",
)


def prepare_archive(source: Path, output: Path) -> dict:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory.")
    players, matches, mapping, saved_strength = (
        pd.read_csv(source / name) for name in SOURCE_NAMES
    )
    require_columns(mapping, ("Squad", "team"))
    if mapping[["Squad", "team"]].isna().any().any() or mapping["Squad"].duplicated().any():
        raise ValueError("The supplied team mapping has missing or duplicate source names.")
    team_map = dict(zip(mapping["Squad"], mapping["team"]))
    prepared_players = prepare_players(players, team_map)
    require_columns(players, ("team",))
    if not prepared_players["team"].equals(players["team"]):
        raise ValueError("Mapped player clubs disagree with the archive's team column.")
    strength = build_team_strength(prepared_players).set_index("team").sort_index()
    require_columns(saved_strength, ("team", *TEAM_FEATURES))
    if saved_strength["team"].duplicated().any():
        raise ValueError("Duplicate teams in saved team_strength.")
    reference = saved_strength.set_index("team").sort_index()
    if not strength.index.equals(reference.index):
        raise ValueError("Rebuilt and supplied team sets differ.")
    np.testing.assert_allclose(
        strength[list(TEAM_FEATURES)], reference[list(TEAM_FEATURES)],
        rtol=1e-10, atol=1e-10,
        err_msg="Rebuilt team averages differ from the supplied archive.",
    )

    base_columns = [f"{side}_{name}" for side in ("home", "away") for name in TEAM_FEATURES]
    require_columns(matches, (*base_columns, "match_id", "competition_name", "target_result"))
    if matches["match_id"].isna().any() or matches["match_id"].duplicated().any():
        raise ValueError("Match IDs must be present and unique.")
    # The archive's train_model.py uses dropna on predictors + target. Differences
    # are missing exactly when their home/away inputs are missing.
    complete = matches[base_columns + ["target_result"]].notna().all(axis=1)
    original_base = [f"{side}_{name}" for side in ("home", "away") for name in (
        *TEAM_FEATURES, "avg_xag", "avg_matches_played", "avg_starts",
    )]
    require_columns(matches, original_base)
    original_complete = matches[original_base + ["target_result"]].notna().all(axis=1)
    if not complete.equals(original_complete):
        raise ValueError("13-feature and original 24-feature completeness masks differ.")
    retained = prepare_matches(matches.loc[complete].copy())
    rebuilt = build_match_features(retained, strength.reset_index())
    np.testing.assert_allclose(
        rebuilt[base_columns], retained[base_columns], rtol=1e-10, atol=1e-10,
        err_msg="Rebuilt match features differ from the supplied archive.",
    )
    excluded = matches.loc[~complete, ["match_id", "competition_name", "home_team", "away_team"]].copy()
    excluded["reason"] = "Missing source match features or target; original complete-case policy"
    report = {
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in SOURCE_NAMES},
        "player_rows": len(players),
        "teams": len(strength),
        "mapping_entries": len(team_map),
        "player_missing_values": {name: int(players[name].isna().sum()) for name in PLAYER_METRICS},
        "matches_total": len(matches),
        "matches_retained": len(retained),
        "matches_excluded": len(excluded),
        "excluded_by_competition": {str(k): int(v) for k, v in excluded["competition_name"].value_counts().items()},
        "retained_class_counts": {str(k): int(v) for k, v in retained["target_result"].value_counts().items()},
        "checks": {
            "mapped_player_teams_match": True,
            "team_means_match_saved_strength": True,
            "match_features_match_saved_dataset": True,
            "same_complete_rows_for_13_and_24_features": True,
            "unique_match_ids": True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    players[["Squad", *PLAYER_METRICS]].to_csv(output / "players.csv", index=False)
    retained[["match_id", "competition_name", "date_utc", "home_team", "away_team", "target_result"]].to_csv(
        output / "matches.csv", index=False,
    )
    excluded.to_csv(output / "excluded_matches.csv", index=False)
    (output / "team_map.json").write_text(json.dumps(team_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original archive's extracted data directory")
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    args = parser.parse_args()
    report = prepare_archive(args.source, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
