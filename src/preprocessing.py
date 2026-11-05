"""Validate user-supplied tables and apply explicit team/target mappings.

Player aggregation uses pandas' column-wise skip-null means, as in Appendix A.
The supplied archive confirms this behavior for its eight missing player ages.
Missing or nonfinite aggregated model features remain errors.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PLAYER_METRICS = ("Gls", "Ast", "xG", "Age", "Min")
CLASSES = ("Home Win", "Draw", "Away Win")


def load_mapping(path: str | Path | None) -> dict[str, str]:
    """Read a JSON object mapping exact source strings to exact destination strings."""
    if path is None:
        return {}
    mapping = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or not k.strip()
        or not isinstance(v, str) or not v.strip()
        for k, v in mapping.items()
    ):
        raise ValueError("Mapping must be a JSON object with nonempty string keys/values.")
    return mapping


def require_columns(frame: pd.DataFrame, columns) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Input table is empty.")


def validate_names(frame: pd.DataFrame, columns) -> None:
    for column in columns:
        valid = frame[column].map(lambda x: isinstance(x, str) and bool(x.strip()))
        if not valid.all():
            raise ValueError(f"Column {column!r} contains missing or empty names.")


def finite_numeric(frame: pd.DataFrame, columns, *, allow_missing=False) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    values = result[list(columns)].to_numpy(dtype=float)
    invalid = np.isinf(values) if allow_missing else ~np.isfinite(values)
    if invalid.any():
        raise ValueError("Numeric features contain missing or infinite values.")
    return result


def prepare_players(players: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    require_columns(players, ("Squad", *PLAYER_METRICS))
    validate_names(players, ("Squad",))
    prepared = finite_numeric(players, PLAYER_METRICS, allow_missing=True)
    # Same mapping direction and unchanged-name fallback as Appendix A.
    prepared["team"] = prepared["Squad"].replace(team_map)
    return prepared


def prepare_matches(
    matches: pd.DataFrame,
    target_column: str = "target_result",
    result_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    require_columns(matches, ("home_team", "away_team", target_column))
    validate_names(matches, ("home_team", "away_team"))
    result = matches.copy()
    labels = result[target_column]
    if labels.isna().any():
        raise ValueError("Target labels contain missing values.")
    labels = labels.astype(str).replace(result_map or {})
    unknown = sorted(set(labels) - set(CLASSES))
    if unknown:
        raise ValueError(
            f"Unknown outcome labels: {unknown}. Supply an explicit --result-map; "
            f"allowed labels: {CLASSES}."
        )
    result["target_result"] = labels
    return result
